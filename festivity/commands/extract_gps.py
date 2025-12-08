"""Extract GPS data from images and populate database."""
import os
import sys
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from datetime import datetime
from PIL import Image, ExifTags
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
from scipy.spatial import cKDTree

from festivity.utils import get_workspace_path, ensure_workspace_initialized, load_config, detect_image_side
from festivity.db import get_db_connection, insert_gps_data, get_filenames_in_gps_data


def register_command(subparsers):
    """Register the extract-gps command."""
    parser = subparsers.add_parser(
        'extract-gps',
        help='Extract GPS data from image EXIF'
    )
    parser.add_argument(
        '--workspace',
        type=str,
        help='Path to workspace directory'
    )
    parser.add_argument(
        '--skip-address-fetch',
        action='store_true',
        help='Skip fetching addresses from OpenStreetMap (default: fetch addresses)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Reprocess all images, even those already in database (default: skip existing)'
    )
    parser.set_defaults(func=execute)


def get_lat_lon(image_path: Path) -> Optional[Tuple[float, float]]:
    """Extract GPS coordinates from image EXIF data."""
    try:
        image = Image.open(image_path)
        
        def decimal_coords(coords, ref):
            decimal_degrees = float(coords[0]) + float(coords[1]) / 60 + float(coords[2]) / 3600
            if ref == "S" or ref == "W":
                decimal_degrees = -1 * decimal_degrees
            return decimal_degrees
        
        GPSINFO_TAG = next(
            tag for tag, name in ExifTags.TAGS.items() if name == "GPSInfo"
        )
        info = image.getexif()
        gpsinfo = info.get_ifd(GPSINFO_TAG)
        
        lat = decimal_coords(gpsinfo[2], gpsinfo[1])
        lon = decimal_coords(gpsinfo[4], gpsinfo[3])
        
        return (lat, lon)
    except Exception:
        return None


def get_timestamp(image_path: Path) -> Optional[datetime]:
    """Extract timestamp from image EXIF data."""
    try:
        image = Image.open(image_path)
        exif = image.getexif()
        
        if exif is None:
            return None
        
        timestamp_tags = ['DateTimeOriginal', 'DateTimeDigitized', 'DateTime']
        
        for tag_name in timestamp_tags:
            tag_id = next((tag for tag, name in ExifTags.TAGS.items() if name == tag_name), None)
            if tag_id and tag_id in exif:
                timestamp_str = exif[tag_id]
                try:
                    return datetime.strptime(timestamp_str, "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    try:
                        return datetime.fromisoformat(timestamp_str)
                    except:
                        continue
        
        return None
    except Exception:
        return None


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two GPS coordinates using Haversine formula."""
    R = 6371000  # Earth's radius in meters
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    
    return distance


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate bearing from point 1 to point 2."""
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)
    
    dlon = lon2_rad - lon1_rad
    
    x = np.sin(dlon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    
    bearing_rad = np.arctan2(x, y)
    bearing_deg = np.degrees(bearing_rad)
    
    return (bearing_deg + 360) % 360


def average_bearings(bearing1: Optional[float], bearing2: Optional[float]) -> Optional[float]:
    """Calculate average of two bearings, handling wrap-around."""
    bearings = [b for b in [bearing1, bearing2] if b is not None]
    
    if not bearings:
        return None
    if len(bearings) == 1:
        return bearings[0]
    
    angles_rad = np.radians(bearings)
    x = np.mean(np.sin(angles_rad))
    y = np.mean(np.cos(angles_rad))
    
    avg_bearing_rad = np.arctan2(x, y)
    avg_bearing_deg = np.degrees(avg_bearing_rad)
    
    return (avg_bearing_deg + 360) % 360


def calculate_headings(df: pd.DataFrame, min_distance_m: float = 2.0) -> pd.DataFrame:
    """Calculate heading for each image based on vectors to previous and next images."""
    headings = []
    
    for idx in range(len(df)):
        bearing_from_prev = None
        bearing_to_next = None
        
        curr_lat = df.iloc[idx]['lat']
        curr_lon = df.iloc[idx]['lon']
        
        # Look backward
        for prev_idx in range(idx - 1, -1, -1):
            prev_lat = df.iloc[prev_idx]['lat']
            prev_lon = df.iloc[prev_idx]['lon']
            distance = calculate_distance(prev_lat, prev_lon, curr_lat, curr_lon)
            
            if distance >= min_distance_m:
                bearing_from_prev = calculate_bearing(prev_lat, prev_lon, curr_lat, curr_lon)
                break
        
        # Look forward
        for next_idx in range(idx + 1, len(df)):
            next_lat = df.iloc[next_idx]['lat']
            next_lon = df.iloc[next_idx]['lon']
            distance = calculate_distance(curr_lat, curr_lon, next_lat, next_lon)
            
            if distance >= min_distance_m:
                bearing_to_next = calculate_bearing(curr_lat, curr_lon, next_lat, next_lon)
                break
        
        avg_heading = average_bearings(bearing_from_prev, bearing_to_next)
        headings.append(avg_heading)
    
    df['heading'] = headings
    return df


def offset_point(lat: float, lon: float, heading: float, distance_m: float, is_left: bool) -> Tuple[float, float]:
    """Calculate a point offset perpendicular to the heading direction."""
    R = 6371000  # Earth's radius in meters
    
    # For left camera, offset to the left (-90 degrees)
    # For right camera, offset to the right (+90 degrees)
    offset_bearing = (heading - 90 if is_left else heading + 90) % 360
    
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    bearing_rad = np.radians(offset_bearing)
    
    new_lat_rad = np.arcsin(
        np.sin(lat_rad) * np.cos(distance_m / R) +
        np.cos(lat_rad) * np.sin(distance_m / R) * np.cos(bearing_rad)
    )
    
    new_lon_rad = lon_rad + np.arctan2(
        np.sin(bearing_rad) * np.sin(distance_m / R) * np.cos(lat_rad),
        np.cos(distance_m / R) - np.sin(lat_rad) * np.sin(new_lat_rad)
    )
    
    new_lat = np.degrees(new_lat_rad)
    new_lon = np.degrees(new_lon_rad)
    
    return (new_lat, new_lon)


def fetch_osm_addresses(min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> List[Dict]:
    """Fetch addresses from OpenStreetMap within the given bounding box."""
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    overpass_query = f"""
    [out:json][timeout:60];
    (
      node["addr:housenumber"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["addr:housenumber"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out center;
    """
    
    print("Fetching addresses from OpenStreetMap...")
    try:
        response = requests.get(overpass_url, params={'data': overpass_query}, timeout=90)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Warning: Error fetching OSM data: {e}")
        return []
    
    addresses = []
    
    for element in data.get('elements', []):
        if element['type'] == 'node':
            lat = element['lat']
            lon = element['lon']
        elif 'center' in element:
            lat = element['center']['lat']
            lon = element['center']['lon']
        else:
            continue
        
        tags = element.get('tags', {})
        addr_parts = []
        
        street_number = tags.get('addr:housenumber', '')
        street_name = tags.get('addr:street', '')
        
        if street_number:
            addr_parts.append(street_number)
        if street_name:
            addr_parts.append(street_name)
        if tags.get('addr:suburb'):
            addr_parts.append(tags['addr:suburb'])
        if tags.get('addr:city'):
            addr_parts.append(tags['addr:city'])
        
        if addr_parts:
            addresses.append({
                'lat': lat,
                'lon': lon,
                'address': ', '.join(addr_parts),
                'street_number': street_number,
                'street_name': street_name
            })
    
    print(f"Found {len(addresses)} addresses from OSM")
    return addresses


def build_address_kdtree(addresses: List[Dict]) -> Tuple[cKDTree, List[Dict]]:
    """Build a KD-tree for fast spatial lookup of addresses.
    
    Returns:
        Tuple of (KD-tree, addresses list)
    """
    if not addresses:
        return None, []
    
    # Extract coordinates for KD-tree (in lat, lon order)
    coords = np.array([[addr['lat'], addr['lon']] for addr in addresses])
    tree = cKDTree(coords)
    return tree, addresses


def match_address_kdtree(offset_lat: float, offset_lon: float, tree: cKDTree, addresses: List[Dict], max_distance_m: float = 30.0) -> Optional[Dict]:
    """Match an offset point to the nearest address using KD-tree for speed.
    
    Note: This uses Euclidean distance in degrees, which is an approximation.
    For the small distances we're searching (30m), this is acceptable.
    """
    if tree is None or not addresses:
        return None
    
    # Convert max distance from meters to approximate degrees
    # At equator: 1 degree latitude ≈ 111,320 meters
    # This is an approximation but fine for our use case
    max_distance_deg = max_distance_m / 111320.0
    
    # Query the tree for nearest neighbor
    distance, idx = tree.query([offset_lat, offset_lon], distance_upper_bound=max_distance_deg)
    
    # Check if we found a valid match
    if idx == len(addresses) or distance == np.inf:
        return None
    
    # Verify with actual haversine distance
    best_addr = addresses[idx]
    actual_distance = calculate_distance(offset_lat, offset_lon, best_addr['lat'], best_addr['lon'])
    
    if actual_distance <= max_distance_m:
        return best_addr
    
    return None


def match_address(offset_lat: float, offset_lon: float, addresses: List[Dict], max_distance_m: float = 30.0) -> Optional[Dict]:
    """Match an offset point to the nearest address within max_distance_m.
    
    DEPRECATED: Use match_address_kdtree for better performance with large address lists.
    """
    if not addresses:
        return None
    
    min_distance = float('inf')
    best_match = None
    
    for addr in addresses:
        distance = calculate_distance(offset_lat, offset_lon, addr['lat'], addr['lon'])
        if distance < min_distance and distance <= max_distance_m:
            min_distance = distance
            best_match = addr
    
    return best_match


def execute(args):
    """Execute the extract-gps command."""
    workspace = get_workspace_path(args)
    ensure_workspace_initialized(workspace)
    
    config = load_config(workspace)
    images_dir = workspace / 'images'
    
    # Get list of images
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f'*{ext}'))
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return 1
    
    # Skip images already in database by default (unless --force)
    if not args.force:
        conn = get_db_connection(workspace)
        existing_filenames = get_filenames_in_gps_data(conn)
        conn.close()
        image_files = [f for f in image_files if f.name not in existing_filenames]
        if not image_files:
            print("All images already processed. Use --force to reprocess.")
            return 0
    
    print(f"\nExtracting GPS data from {len(image_files)} images...")
    
    # Extract GPS coordinates and timestamps
    results = []
    skipped = 0
    
    for img_path in tqdm(image_files, desc="Reading EXIF data"):
        coords = get_lat_lon(img_path)
        if coords is None:
            skipped += 1
            continue
        
        lat, lon = coords
        timestamp = get_timestamp(img_path)
        is_left = detect_image_side(img_path.name, config)
        
        results.append({
            'filename': img_path.name,
            'lat': lat,
            'lon': lon,
            'timestamp': timestamp,
            'is_left': is_left
        })
    
    if not results:
        print(f"No GPS data found in any images ({skipped} images skipped)")
        return 1
    
    print(f"✓ Extracted GPS data from {len(results)} images ({skipped} skipped)")
    
    # Create DataFrame and sort by timestamp
    df = pd.DataFrame(results)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate headings separately for left and right cameras
    print("\nCalculating headings...")
    
    # Split into left and right cameras
    left_df = df[df['is_left'] == True].copy()
    right_df = df[df['is_left'] == False].copy()
    unknown_df = df[df['is_left'].isna()].copy()
    
    # Calculate headings for each camera separately
    if len(left_df) > 0:
        left_df = calculate_headings(left_df, min_distance_m=2.0)
    if len(right_df) > 0:
        right_df = calculate_headings(right_df, min_distance_m=2.0)
    if len(unknown_df) > 0:
        unknown_df = calculate_headings(unknown_df, min_distance_m=2.0)
    
    # Combine back together
    df = pd.concat([left_df, right_df, unknown_df], ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate offset points
    print("Calculating offset points...")
    # Support both old and new config formats for backward compatibility
    offset_distance_left = config.get('offset_distance_left_m', 
                                      config.get('camera_offset_distance_m', 
                                                config.get('offset_distance_m', 30.0)))
    offset_distance_right = config.get('offset_distance_right_m',
                                       config.get('camera_offset_distance_m',
                                                 config.get('offset_distance_m', 20.0)))
    
    offset_lats = []
    offset_lons = []
    
    for idx, row in df.iterrows():
        if pd.notna(row['heading']) and row['is_left'] is not None:
            # Use appropriate offset distance based on camera side
            offset_distance = offset_distance_left if row['is_left'] else offset_distance_right
            offset_lat, offset_lon = offset_point(
                row['lat'], row['lon'], row['heading'], offset_distance, row['is_left']
            )
            offset_lats.append(offset_lat)
            offset_lons.append(offset_lon)
        else:
            offset_lats.append(None)
            offset_lons.append(None)
    
    df['offset_lat'] = offset_lats
    df['offset_lon'] = offset_lons
    
    # Fetch and match addresses (default behavior, unless skipped)
    if not args.skip_address_fetch:
        min_lat = df['offset_lat'].min()
        max_lat = df['offset_lat'].max()
        min_lon = df['offset_lon'].min()
        max_lon = df['offset_lon'].max()
        
        addresses = fetch_osm_addresses(min_lat, max_lat, min_lon, max_lon)
        
        # Build KD-tree for fast address matching
        print("Building spatial index for addresses...")
        tree, addresses = build_address_kdtree(addresses)
        
        print("Matching addresses...")
        matched_data = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Matching addresses"):
            if pd.notna(row['offset_lat']):
                match = match_address_kdtree(row['offset_lat'], row['offset_lon'], tree, addresses)
                if match:
                    matched_data.append({
                        'address': match['address'],
                        'street_number': match['street_number'],
                        'street_name': match['street_name'],
                        'address_lat': match['lat'],
                        'address_lon': match['lon']
                    })
                else:
                    matched_data.append({
                        'address': None,
                        'street_number': None,
                        'street_name': None,
                        'address_lat': None,
                        'address_lon': None
                    })
            else:
                matched_data.append({
                    'address': None,
                    'street_number': None,
                    'street_name': None,
                    'address_lat': None,
                    'address_lon': None
                })
        
        matched_df = pd.DataFrame(matched_data)
        df = pd.concat([df, matched_df], axis=1)
    
    # Insert into database
    print("\nSaving to database...")
    conn = get_db_connection(workspace)
    
    # Convert DataFrame to list of dicts for bulk insert
    gps_records = []
    for idx, row in df.iterrows():
        record = {
            'filename': row['filename'],
            'is_left': row['is_left'],
            'lat': row['lat'],
            'lon': row['lon'],
            'timestamp': row['timestamp'].isoformat() if pd.notna(row['timestamp']) else None,
            'heading': row['heading'] if pd.notna(row['heading']) else None,
            'offset_lat': row['offset_lat'] if pd.notna(row['offset_lat']) else None,
            'offset_lon': row['offset_lon'] if pd.notna(row['offset_lon']) else None,
            'address': row.get('address') if pd.notna(row.get('address')) else None,
            'street_number': row.get('street_number') if pd.notna(row.get('street_number')) else None,
            'street_name': row.get('street_name') if pd.notna(row.get('street_name')) else None,
            'address_lat': row.get('address_lat') if pd.notna(row.get('address_lat')) else None,
            'address_lon': row.get('address_lon') if pd.notna(row.get('address_lon')) else None,
        }
        gps_records.append(record)
    
    insert_gps_data(conn, gps_records)
    conn.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("GPS Extraction Summary")
    print("=" * 60)
    print(f"  Total images processed: {len(results)}")
    print(f"  Images with GPS data: {len(results)}")
    print(f"  Images skipped (no GPS): {skipped}")
    print(f"  Headings calculated: {df['heading'].notna().sum()}")
    print(f"  Offset points calculated: {df['offset_lat'].notna().sum()}")
    
    if not args.skip_address_fetch:
        matched_count = df['address'].notna().sum() if 'address' in df.columns else 0
        # Count unique addresses (excluding None, 'none', and empty strings)
        unique_addresses = df[df['address'].notna() & (df['address'] != 'none') & (df['address'] != '')]['address'].nunique()
        print(f"  Addresses matched: {matched_count}")
        print(f"  Unique addresses: {unique_addresses}")
    
    print(f"\nNext step:")
    print(f"  festivity score --workspace {workspace}")
    
    return 0
