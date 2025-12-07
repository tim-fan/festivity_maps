"""Trajectory statistics calculations."""
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import math
import sqlite3


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
    
    Returns:
        Distance in meters
    """
    # Earth radius in meters
    R = 6371000
    
    # Convert to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


def calculate_trajectory_stats(db_path: Path, gap_threshold_seconds: float = 10.0, min_distance_m: float = 2.0) -> Dict:
    """
    Calculate trajectory statistics from GPS data.
    
    Calculates trajectories separately for left and right cameras (since they
    record simultaneously at different positions), then averages the distances.
    
    Splits trajectory into sub-trajectories when timestamp gaps exceed threshold.
    Only counts distance increments above min_distance_m to filter GPS noise.
    
    Args:
        db_path: Path to database file
        gap_threshold_seconds: Maximum gap between consecutive images before
                              treating as a new sub-trajectory (default: 10 seconds)
        min_distance_m: Minimum distance between consecutive points to count
                       toward trajectory length (default: 2.0 meters)
    
    Returns:
        Dictionary containing:
            - total_images: Total number of images with GPS data
            - total_distance_m: Average trajectory distance in meters (left + right) / 2
            - total_duration_seconds: Total trajectory duration in seconds
            - num_sub_trajectories: Number of sub-trajectories detected
            - left_distance_m: Left camera trajectory distance
            - right_distance_m: Right camera trajectory distance
            - sub_trajectories: List of sub-trajectory info dicts
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Get GPS data for left camera
    cursor.execute("""
        SELECT filename, lat, lon, timestamp
        FROM gps_data
        WHERE lat IS NOT NULL AND lon IS NOT NULL AND timestamp IS NOT NULL
        AND is_left = 1
        ORDER BY timestamp
    """)
    left_rows = cursor.fetchall()
    
    # Get GPS data for right camera
    cursor.execute("""
        SELECT filename, lat, lon, timestamp
        FROM gps_data
        WHERE lat IS NOT NULL AND lon IS NOT NULL AND timestamp IS NOT NULL
        AND is_left = 0
        ORDER BY timestamp
    """)
    right_rows = cursor.fetchall()
    
    conn.close()
    
    if not left_rows and not right_rows:
        return {
            'total_images': 0,
            'total_distance_m': 0.0,
            'total_duration_seconds': 0.0,
            'num_sub_trajectories': 0,
            'left_distance_m': 0.0,
            'right_distance_m': 0.0,
            'sub_trajectories': []
        }
    
    # Calculate trajectory for left camera
    left_stats = _calculate_single_trajectory(left_rows, gap_threshold_seconds, min_distance_m, 'left')
    
    # Calculate trajectory for right camera
    right_stats = _calculate_single_trajectory(right_rows, gap_threshold_seconds, min_distance_m, 'right')
    
    # Average the distances (both cameras recording same trajectory)
    total_distance = (left_stats['distance_m'] + right_stats['distance_m']) / 2.0
    
    # Use the maximum duration (should be similar for both)
    total_duration = max(left_stats['duration_seconds'], right_stats['duration_seconds'])
    
    # Combine images from both cameras
    total_images = len(left_rows) + len(right_rows)
    
    # Merge sub-trajectories from left and right cameras by time overlap
    merged_subs = _merge_sub_trajectories(left_stats['sub_trajectories'], right_stats['sub_trajectories'])
    
    return {
        'total_images': total_images,
        'total_distance_m': total_distance,
        'total_duration_seconds': total_duration,
        'num_sub_trajectories': len(merged_subs),
        'sub_trajectories': merged_subs
    }


def _merge_sub_trajectories(left_subs: List[Dict], right_subs: List[Dict]) -> List[Dict]:
    """
    Merge left and right camera sub-trajectories that overlap in time.
    
    Args:
        left_subs: Sub-trajectories from left camera
        right_subs: Sub-trajectories from right camera
    
    Returns:
        List of merged sub-trajectories with averaged distances
    """
    if not left_subs and not right_subs:
        return []
    
    # Combine and sort by start time
    all_subs = []
    for sub in left_subs:
        all_subs.append(('left', sub))
    for sub in right_subs:
        all_subs.append(('right', sub))
    
    all_subs.sort(key=lambda x: x[1]['start_time'])
    
    # Group by time overlap
    merged = []
    current_group = {'left': None, 'right': None}
    
    for camera, sub in all_subs:
        current_group[camera] = sub
        
        # If we have both left and right for this time period, merge them
        if current_group['left'] and current_group['right']:
            left_sub = current_group['left']
            right_sub = current_group['right']
            
            merged.append({
                'images': left_sub['images'] + right_sub['images'],
                'distance_m': (left_sub['distance_m'] + right_sub['distance_m']) / 2.0,
                'duration_seconds': max(left_sub['duration_seconds'], right_sub['duration_seconds']),
                'start_time': min(left_sub['start_time'], right_sub['start_time']),
                'end_time': max(left_sub['end_time'], right_sub['end_time'])
            })
            
            # Reset for next group
            current_group = {'left': None, 'right': None}
    
    # Handle any remaining unpaired sub-trajectory
    if current_group['left'] or current_group['right']:
        remaining = current_group['left'] or current_group['right']
        merged.append({
            'images': remaining['images'],
            'distance_m': remaining['distance_m'],
            'duration_seconds': remaining['duration_seconds'],
            'start_time': remaining['start_time'],
            'end_time': remaining['end_time']
        })
    
    return merged


def _calculate_single_trajectory(rows: List[Tuple], gap_threshold_seconds: float, min_distance_m: float, camera_name: str) -> Dict:
    """
    Calculate trajectory statistics for a single camera.
    
    Args:
        rows: List of (filename, lat, lon, timestamp) tuples
        gap_threshold_seconds: Maximum gap between consecutive images
        min_distance_m: Minimum distance to count toward trajectory length
        camera_name: Name of camera ('left' or 'right') for labeling
    
    Returns:
        Dictionary with distance_m, duration_seconds, and sub_trajectories
    """
    if not rows:
        return {
            'distance_m': 0.0,
            'duration_seconds': 0.0,
            'sub_trajectories': []
        }
    
    total_distance = 0.0
    total_duration = 0.0
    sub_trajectories = []
    
    # Start first sub-trajectory
    current_sub = {
        'camera': camera_name,
        'start_idx': 0,
        'images': 1,
        'distance_m': 0.0,
        'duration_seconds': 0.0,
        'start_time': datetime.fromisoformat(rows[0][3]),
        'end_time': datetime.fromisoformat(rows[0][3])
    }
    
    prev_lat, prev_lon = rows[0][1], rows[0][2]
    prev_time = datetime.fromisoformat(rows[0][3])
    
    for i in range(1, len(rows)):
        filename, lat, lon, timestamp = rows[i]
        current_time = datetime.fromisoformat(timestamp)
        
        # Check for gap
        time_gap = (current_time - prev_time).total_seconds()
        
        if time_gap > gap_threshold_seconds:
            # End current sub-trajectory
            sub_trajectories.append(current_sub)
            
            # Start new sub-trajectory
            current_sub = {
                'camera': camera_name,
                'start_idx': i,
                'images': 1,
                'distance_m': 0.0,
                'duration_seconds': 0.0,
                'start_time': current_time,
                'end_time': current_time
            }
            # Reset prev position for new sub-trajectory
            prev_lat, prev_lon = lat, lon
            prev_time = current_time
        else:
            # Continue current sub-trajectory
            distance = haversine_distance(prev_lat, prev_lon, lat, lon)
            
            # Only count distance if it exceeds minimum threshold (filters GPS noise)
            if distance >= min_distance_m:
                current_sub['distance_m'] += distance
                total_distance += distance
                # Update prev position only when we count the distance
                prev_lat, prev_lon = lat, lon
            
            # Always count time and images
            current_sub['duration_seconds'] += time_gap
            current_sub['images'] += 1
            current_sub['end_time'] = current_time
            total_duration += time_gap
            prev_time = current_time
    
    # Add final sub-trajectory
    sub_trajectories.append(current_sub)
    
    return {
        'distance_m': total_distance,
        'duration_seconds': total_duration,
        'sub_trajectories': sub_trajectories
    }


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted string like "2h 34m" or "45m 12s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def format_distance(meters: float) -> str:
    """
    Format distance in meters to human-readable string with imperial in brackets.
    
    Args:
        meters: Distance in meters
    
    Returns:
        Formatted string like "12.3 km (7.6 mi)" or "456 m (1,497 ft)"
    """
    # Convert to miles and feet
    miles = meters * 0.000621371
    feet = meters * 3.28084
    
    if meters >= 1000:
        km = meters / 1000
        return f"{km:.2f} km ({miles:.2f} mi)"
    else:
        return f"{meters:.0f} m ({feet:.0f} ft)"


def format_speed(meters_per_second: float) -> str:
    """
    Format speed in m/s to human-readable string with imperial in brackets.
    
    Args:
        meters_per_second: Speed in meters per second
    
    Returns:
        Formatted string like "12.5 km/h (7.8 mph)"
    """
    kmh = meters_per_second * 3.6
    mph = meters_per_second * 2.23694
    return f"{kmh:.1f} km/h ({mph:.1f} mph)"
