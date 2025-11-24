#!/usr/bin/env python3
"""
Extract GPS coordinates (latitude/longitude) from geotagged images.

This script processes images in a directory, extracts GPS EXIF data,
and saves the results to a CSV file. Optionally displays the locations
on an interactive map.
"""

import argparse
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from datetime import datetime
import time
from PIL import Image, ExifTags
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests


def get_lat_lon(image_path: Path) -> Optional[Tuple[float, float]]:
    """
    Extract GPS coordinates from image EXIF data.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (latitude, longitude) or None if GPS data not found
    """
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
    except Exception as e:
        return None


def get_timestamp(image_path: Path) -> Optional[datetime]:
    """
    Extract timestamp from image EXIF data.
    
    Tries multiple EXIF fields in order of preference:
    1. DateTimeOriginal - when the photo was taken
    2. DateTimeDigitized - when the photo was digitized
    3. DateTime - when the file was modified
    
    Args:
        image_path: Path to the image file
        
    Returns:
        datetime object or None if timestamp not found
    """
    try:
        image = Image.open(image_path)
        exif = image.getexif()
        
        if exif is None:
            return None
        
        # Try different timestamp fields in order of preference
        timestamp_tags = ['DateTimeOriginal', 'DateTimeDigitized', 'DateTime']
        
        for tag_name in timestamp_tags:
            tag_id = next((tag for tag, name in ExifTags.TAGS.items() if name == tag_name), None)
            if tag_id and tag_id in exif:
                timestamp_str = exif[tag_id]
                # EXIF datetime format is typically "YYYY:MM:DD HH:MM:SS"
                try:
                    return datetime.strptime(timestamp_str, "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    # Try alternative format
                    try:
                        return datetime.fromisoformat(timestamp_str)
                    except:
                        continue
        
        return None
    except Exception as e:
        return None


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the distance between two GPS coordinates using the Haversine formula.
    
    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
        
    Returns:
        Distance in meters
    """
    # Earth's radius in meters
    R = 6371000
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    
    return distance


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the bearing (heading) from point 1 to point 2.
    
    Uses the forward azimuth formula for calculating bearing between two GPS coordinates.
    
    Args:
        lat1, lon1: Starting point coordinates (degrees)
        lat2, lon2: Ending point coordinates (degrees)
        
    Returns:
        Bearing in degrees from north (0-360)
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)
    
    # Calculate bearing
    dlon = lon2_rad - lon1_rad
    
    x = np.sin(dlon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    
    bearing_rad = np.arctan2(x, y)
    bearing_deg = np.degrees(bearing_rad)
    
    # Normalize to 0-360
    bearing_deg = (bearing_deg + 360) % 360
    
    return bearing_deg


def average_bearings(bearing1: Optional[float], bearing2: Optional[float]) -> Optional[float]:
    """
    Calculate the average of two bearings, handling wrap-around at 0/360 degrees.
    
    Uses circular mean to properly average angles.
    
    Args:
        bearing1: First bearing in degrees (0-360) or None
        bearing2: Second bearing in degrees (0-360) or None
        
    Returns:
        Average bearing in degrees (0-360) or None if both inputs are None
    """
    bearings = [b for b in [bearing1, bearing2] if b is not None]
    
    if not bearings:
        return None
    
    if len(bearings) == 1:
        return bearings[0]
    
    # Convert to unit vectors and average
    angles_rad = np.radians(bearings)
    x = np.mean(np.sin(angles_rad))
    y = np.mean(np.cos(angles_rad))
    
    avg_bearing_rad = np.arctan2(x, y)
    avg_bearing_deg = np.degrees(avg_bearing_rad)
    
    # Normalize to 0-360
    avg_bearing_deg = (avg_bearing_deg + 360) % 360
    
    return avg_bearing_deg


def calculate_headings(df: pd.DataFrame, min_distance_m: float = 2.0) -> pd.DataFrame:
    """
    Calculate heading for each image based on vectors to previous and next images.
    
    To reduce heading noise from GPS position errors when stationary, only uses
    reference points that are at least min_distance_m away from the current point.
    
    Args:
        df: DataFrame with lat, lon columns (must be sorted by timestamp)
        min_distance_m: Minimum distance in meters to reference point for heading calculation
        
    Returns:
        DataFrame with added 'heading' column
    """
    headings = []
    
    for idx in range(len(df)):
        bearing_from_prev = None
        bearing_to_next = None
        
        curr_lat = df.iloc[idx]['lat']
        curr_lon = df.iloc[idx]['lon']
        
        # Look backward to find a point at least min_distance_m away
        for prev_idx in range(idx - 1, -1, -1):
            prev_lat = df.iloc[prev_idx]['lat']
            prev_lon = df.iloc[prev_idx]['lon']
            distance = calculate_distance(prev_lat, prev_lon, curr_lat, curr_lon)
            
            if distance >= min_distance_m:
                bearing_from_prev = calculate_bearing(prev_lat, prev_lon, curr_lat, curr_lon)
                break
        
        # Look forward to find a point at least min_distance_m away
        for next_idx in range(idx + 1, len(df)):
            next_lat = df.iloc[next_idx]['lat']
            next_lon = df.iloc[next_idx]['lon']
            distance = calculate_distance(curr_lat, curr_lon, next_lat, next_lon)
            
            if distance >= min_distance_m:
                bearing_to_next = calculate_bearing(curr_lat, curr_lon, next_lat, next_lon)
                break
        
        # Average the two bearings
        avg_heading = average_bearings(bearing_from_prev, bearing_to_next)
        headings.append(avg_heading)
    
    df['heading'] = headings
    
    return df


def offset_point(lat: float, lon: float, heading: float, distance_m: float) -> Tuple[float, float]:
    """
    Calculate a point offset perpendicular to the heading direction.
    
    The offset is to the right of the heading direction (i.e., -90 degrees from heading).
    
    Args:
        lat: Latitude of the original point (degrees)
        lon: Longitude of the original point (degrees)
        heading: Heading in degrees from north (0-360)
        distance_m: Distance to offset in meters (positive = right side)
        
    Returns:
        Tuple of (offset_lat, offset_lon)
    """
    # Earth's radius in meters
    R = 6371000
    
    # Calculate the perpendicular bearing (90 degrees to the right)
    offset_bearing = (heading - 90) % 360
    
    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    bearing_rad = np.radians(offset_bearing)
    
    # Calculate the new position using the haversine formula
    new_lat_rad = np.arcsin(
        np.sin(lat_rad) * np.cos(distance_m / R) +
        np.cos(lat_rad) * np.sin(distance_m / R) * np.cos(bearing_rad)
    )
    
    new_lon_rad = lon_rad + np.arctan2(
        np.sin(bearing_rad) * np.sin(distance_m / R) * np.cos(lat_rad),
        np.cos(distance_m / R) - np.sin(lat_rad) * np.sin(new_lat_rad)
    )
    
    # Convert back to degrees
    new_lat = np.degrees(new_lat_rad)
    new_lon = np.degrees(new_lon_rad)
    
    return (new_lat, new_lon)


def calculate_offset_points(df: pd.DataFrame, offset_distance_m: float) -> pd.DataFrame:
    """
    Calculate offset points perpendicular to the heading direction.
    
    Args:
        df: DataFrame with lat, lon, and heading columns
        offset_distance_m: Distance to offset in meters (positive = right/east of heading)
        
    Returns:
        DataFrame with added 'offset_lat' and 'offset_lon' columns
    """
    offset_lats = []
    offset_lons = []
    
    for idx, row in df.iterrows():
        if pd.notna(row['heading']):
            offset_lat, offset_lon = offset_point(
                row['lat'], row['lon'], row['heading'], offset_distance_m
            )
            offset_lats.append(offset_lat)
            offset_lons.append(offset_lon)
        else:
            offset_lats.append(None)
            offset_lons.append(None)
    
    df['offset_lat'] = offset_lats
    df['offset_lon'] = offset_lons
    
    return df


def fetch_osm_addresses(min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> List[Dict]:
    """
    Fetch addresses from OpenStreetMap within the given bounding box.
    
    Uses the Overpass API to query for buildings and addresses.
    
    Args:
        min_lat, max_lat: Latitude bounds
        min_lon, max_lon: Longitude bounds
        
    Returns:
        List of dictionaries with 'lat', 'lon', 'address' keys
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Overpass QL query for buildings with addresses
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
        error_msg = f"Error fetching OSM data: {e}"
        print(error_msg)
        raise RuntimeError(error_msg) from e
    
    addresses = []
    
    for element in data.get('elements', []):
        # Get coordinates
        if element['type'] == 'node':
            lat = element['lat']
            lon = element['lon']
        elif 'center' in element:
            lat = element['center']['lat']
            lon = element['center']['lon']
        else:
            continue
        
        # Build address string
        tags = element.get('tags', {})
        addr_parts = []
        
        if 'addr:housenumber' in tags:
            addr_parts.append(tags['addr:housenumber'])
        if 'addr:street' in tags:
            addr_parts.append(tags['addr:street'])
        if 'addr:suburb' in tags:
            addr_parts.append(tags['addr:suburb'])
        if 'addr:city' in tags:
            addr_parts.append(tags['addr:city'])
        
        if addr_parts:
            address_str = ', '.join(addr_parts)
            addresses.append({
                'lat': lat,
                'lon': lon,
                'address': address_str
            })
    
    print(f"Found {len(addresses)} addresses from OSM")
    return addresses


def match_addresses_to_points(df: pd.DataFrame, addresses: List[Dict], max_distance_m: float = 30.0) -> pd.DataFrame:
    """
    Match offset points to nearest addresses within max_distance_m.
    
    Args:
        df: DataFrame with offset_lat, offset_lon columns
        addresses: List of address dictionaries from OSM
        max_distance_m: Maximum distance in meters to consider a match
        
    Returns:
        DataFrame with added 'address', 'address_lat', 'address_lon' columns
    """
    if not addresses:
        # No addresses available, use defaults
        df['address'] = 'none'
        df['address_lat'] = df['offset_lat']
        df['address_lon'] = df['offset_lon']
        return df
    
    # Convert addresses to numpy arrays for vectorized distance calculations
    addr_lats = np.array([a['lat'] for a in addresses])
    addr_lons = np.array([a['lon'] for a in addresses])
    addr_strings = [a['address'] for a in addresses]
    
    matched_addresses = []
    matched_lats = []
    matched_lons = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Matching addresses"):
        if pd.isna(row['offset_lat']) or pd.isna(row['offset_lon']):
            matched_addresses.append('none')
            matched_lats.append(row['offset_lat'])
            matched_lons.append(row['offset_lon'])
            continue
        
        # Calculate distances to all addresses
        distances = np.array([
            calculate_distance(row['offset_lat'], row['offset_lon'], addr_lat, addr_lon)
            for addr_lat, addr_lon in zip(addr_lats, addr_lons)
        ])
        
        # Find closest address
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        if min_distance <= max_distance_m:
            # Found a match
            matched_addresses.append(addr_strings[min_idx])
            matched_lats.append(addr_lats[min_idx])
            matched_lons.append(addr_lons[min_idx])
        else:
            # No match within threshold
            matched_addresses.append('none')
            matched_lats.append(row['offset_lat'])
            matched_lons.append(row['offset_lon'])
    
    df['address'] = matched_addresses
    df['address_lat'] = matched_lats
    df['address_lon'] = matched_lons
    
    return df


def process_images(image_dir: Path, output_csv: Path, offset_distance_m: float = 10.0, 
                   fetch_addresses: bool = False, verbose: bool = False):
    """
    Process all images in a directory and extract GPS coordinates.
    
    Args:
        image_dir: Directory containing images
        output_csv: Path to output CSV file
        offset_distance_m: Distance in meters to offset points perpendicular to heading
        fetch_addresses: Whether to fetch and match addresses from OSM
        verbose: Whether to print detailed progress
    """
    # Support common image extensions
    image_extensions = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(image_dir.glob(ext))
    
    image_paths = sorted(image_paths)
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    results = []
    skipped = 0
    
    for image_path in tqdm(image_paths, desc="Extracting GPS data"):
        coords = get_lat_lon(image_path)
        timestamp = get_timestamp(image_path)
        
        if coords is not None:
            lat, lon = coords
            result = {
                "filename": image_path.name,
                "lat": lat,
                "lon": lon,
                "timestamp": timestamp,
                "path": str(image_path.absolute())
            }
            results.append(result)
        else:
            skipped += 1
            if verbose:
                print(f"No GPS data found in {image_path.name}")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    
    # Sort by timestamp (images without timestamp will be at the end)
    if not df.empty and 'timestamp' in df.columns:
        df = df.sort_values('timestamp', na_position='last')
        # Reset index after sorting
        df = df.reset_index(drop=True)
    
    # Calculate headings based on GPS trajectory
    if not df.empty and len(df) > 1:
        df = calculate_headings(df)
    else:
        df['heading'] = None
    
    # Calculate offset points
    if not df.empty:
        df = calculate_offset_points(df, offset_distance_m)
    else:
        df['offset_lat'] = None
        df['offset_lon'] = None
    
    # Fetch and match addresses if requested
    if fetch_addresses and not df.empty:
        # Get bounding box
        min_lat = df['lat'].min()
        max_lat = df['lat'].max()
        min_lon = df['lon'].min()
        max_lon = df['lon'].max()
        
        # Add some padding to bounding box (about 0.001 degrees ~= 100m)
        padding = 0.001
        min_lat -= padding
        max_lat += padding
        min_lon -= padding
        max_lon += padding
        
        # Fetch addresses from OSM
        addresses = fetch_osm_addresses(min_lat, max_lat, min_lon, max_lon)
        
        # Match addresses to offset points
        df = match_addresses_to_points(df, addresses)
    else:
        # Add default address columns
        df['address'] = 'none'
        df['address_lat'] = df['offset_lat']
        df['address_lon'] = df['offset_lon']
    
    # Create output directory if it doesn't exist
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_csv, index=False)
    print(f"\nProcessed {len(results)} images with GPS data")
    print(f"Skipped {skipped} images without GPS data")
    print(f"Results saved to {output_csv}")
    
    return df


def visualize_on_map(csv_path: Path, output_html: Optional[Path] = None):
    """
    Create an interactive map visualization of the GPS coordinates.
    
    Uses folium to create an HTML map with markers for original and offset locations.
    Includes toggleable layers for different visualizations.
    
    Args:
        csv_path: Path to the CSV file with GPS data
        output_html: Path to save the HTML map (default: same name as CSV with .html extension)
    """
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        print("Error: folium is required for map visualization")
        print("Install it with: pip install folium")
        return
    
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("No data to visualize")
        return
    
    # Calculate center of map
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()
    
    # Create map with layer control
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    # Create feature groups for different layers
    original_markers = folium.FeatureGroup(name='Original Points', show=True)
    offset_markers = folium.FeatureGroup(name='Offset Points', show=True)
    address_markers = folium.FeatureGroup(name='Address Points', show=False)
    original_heatmap_layer = folium.FeatureGroup(name='Original Heatmap', show=False)
    offset_heatmap_layer = folium.FeatureGroup(name='Offset Heatmap', show=False)
    
    # Add markers for original points (blue)
    for idx, row in df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            popup=f"{row['filename']}<br>Heading: {row.get('heading', 'N/A'):.1f}°" if pd.notna(row.get('heading')) else row['filename'],
            tooltip=row["filename"],
            color='blue',
            fill=True,
            fillColor='blue',
            fillOpacity=0.6
        ).add_to(original_markers)
    
    # Add markers for offset points (red)
    if 'offset_lat' in df.columns and 'offset_lon' in df.columns:
        for idx, row in df.iterrows():
            if pd.notna(row['offset_lat']) and pd.notna(row['offset_lon']):
                folium.CircleMarker(
                    location=[row["offset_lat"], row["offset_lon"]],
                    radius=5,
                    popup=f"{row['filename']} (offset)<br>Heading: {row.get('heading', 'N/A'):.1f}°" if pd.notna(row.get('heading')) else f"{row['filename']} (offset)",
                    tooltip=f"{row['filename']} (offset)",
                    color='red',
                    fill=True,
                    fillColor='red',
                    fillOpacity=0.6
                ).add_to(offset_markers)
    
    # Add markers for matched address points (green)
    if 'address_lat' in df.columns and 'address_lon' in df.columns and 'address' in df.columns:
        # Only show addresses that are actually matched (not default offset points)
        matched_addresses = {}
        for idx, row in df.iterrows():
            if (pd.notna(row['address_lat']) and pd.notna(row['address_lon']) and 
                row.get('address', 'none') != 'none'):
                # Use address location as key to avoid duplicate markers
                addr_key = (round(row['address_lat'], 6), round(row['address_lon'], 6))
                if addr_key not in matched_addresses:
                    matched_addresses[addr_key] = {
                        'lat': row['address_lat'],
                        'lon': row['address_lon'],
                        'address': row['address'],
                        'images': []
                    }
                matched_addresses[addr_key]['images'].append(row['filename'])
        
        # Add markers for unique addresses
        for addr_data in matched_addresses.values():
            image_list = '<br>'.join(addr_data['images'][:5])  # Show first 5 images
            if len(addr_data['images']) > 5:
                image_list += f"<br>... and {len(addr_data['images']) - 5} more"
            
            folium.CircleMarker(
                location=[addr_data['lat'], addr_data['lon']],
                radius=6,
                popup=f"<b>{addr_data['address']}</b><br><br>Images ({len(addr_data['images'])}):<br>{image_list}",
                tooltip=addr_data['address'],
                color='green',
                fill=True,
                fillColor='green',
                fillOpacity=0.7
            ).add_to(address_markers)
    
    # Add heatmap for original points
    if len(df) > 2:
        heat_data_original = [[row["lat"], row["lon"]] for idx, row in df.iterrows()]
        HeatMap(heat_data_original, name='Original Heatmap').add_to(original_heatmap_layer)
    
    # Add heatmap for offset points
    if 'offset_lat' in df.columns and 'offset_lon' in df.columns:
        heat_data_offset = [
            [row["offset_lat"], row["offset_lon"]] 
            for idx, row in df.iterrows() 
            if pd.notna(row['offset_lat']) and pd.notna(row['offset_lon'])
        ]
        if len(heat_data_offset) > 2:
            HeatMap(heat_data_offset, name='Offset Heatmap').add_to(offset_heatmap_layer)
    
    # Add all feature groups to map
    original_markers.add_to(m)
    offset_markers.add_to(m)
    address_markers.add_to(m)
    original_heatmap_layer.add_to(m)
    offset_heatmap_layer.add_to(m)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Determine output path
    if output_html is None:
        output_html = csv_path.with_suffix(".html")
    
    # Save map
    m.save(str(output_html))
    print(f"\nMap saved to {output_html}")
    print(f"Open this file in a web browser to view the interactive map")
    print(f"Use the layer control in the top-right corner to toggle different visualizations")


def main():
    parser = argparse.ArgumentParser(
        description="Extract GPS coordinates from geotagged images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - extract GPS and save to CSV
  %(prog)s /path/to/images output.csv
  
  # Extract GPS with custom offset distance
  %(prog)s /path/to/images output.csv --offset 15.0
  
  # Fetch and match addresses from OpenStreetMap
  %(prog)s /path/to/images output.csv --fetch-addresses
  
  # Extract GPS and view on map
  %(prog)s /path/to/images output.csv --view-map
  
  # Specify custom map output location
  %(prog)s /path/to/images output.csv --view-map --map-output my_map.html
        """
    )
    
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing geotagged images"
    )
    
    parser.add_argument(
        "output_csv",
        type=Path,
        help="Output CSV file path (columns: filename, lat, lon, timestamp, heading, offset_lat, offset_lon, address, address_lat, address_lon)"
    )
    
    parser.add_argument(
        "--offset",
        type=float,
        default=-20.0,
        help="Distance in meters to offset points perpendicular to heading (default: -20.0). Positive = right side of heading."
    )
    
    parser.add_argument(
        "--fetch-addresses",
        action="store_true",
        help="Fetch addresses from OpenStreetMap and match to offset points (within 30m)"
    )
    
    parser.add_argument(
        "--view-map",
        action="store_true",
        help="Generate and view an interactive HTML map of the locations"
    )
    
    parser.add_argument(
        "--map-output",
        type=Path,
        help="Custom output path for the HTML map (default: same as CSV with .html extension)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    # Check if image directory exists
    if not args.image_dir.exists():
        print(f"Error: Image directory '{args.image_dir}' does not exist")
        return 1
    
    if not args.image_dir.is_dir():
        print(f"Error: '{args.image_dir}' is not a directory")
        return 1
    
    # Process images
    df = process_images(args.image_dir, args.output_csv, args.offset, args.fetch_addresses, args.verbose)
    
    # Optionally create map visualization
    if args.view_map and df is not None and not df.empty:
        visualize_on_map(args.output_csv, args.map_output)
    
    return 0


if __name__ == "__main__":
    exit(main())
