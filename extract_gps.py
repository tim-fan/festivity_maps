#!/usr/bin/env python3
"""
Extract GPS coordinates (latitude/longitude) from geotagged images.

This script processes images in a directory, extracts GPS EXIF data,
and saves the results to a CSV file. Optionally displays the locations
on an interactive map.
"""

import argparse
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime
from PIL import Image, ExifTags
import pandas as pd
import numpy as np
from tqdm import tqdm


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


def calculate_headings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate heading for each image based on vectors to previous and next images.
    
    Args:
        df: DataFrame with lat, lon columns (must be sorted by timestamp)
        
    Returns:
        DataFrame with added 'heading' column
    """
    headings = []
    
    for idx in range(len(df)):
        bearing_from_prev = None
        bearing_to_next = None
        
        # Calculate bearing from previous image to current
        if idx > 0:
            prev_lat = df.iloc[idx - 1]['lat']
            prev_lon = df.iloc[idx - 1]['lon']
            curr_lat = df.iloc[idx]['lat']
            curr_lon = df.iloc[idx]['lon']
            bearing_from_prev = calculate_bearing(prev_lat, prev_lon, curr_lat, curr_lon)
        
        # Calculate bearing from current image to next
        if idx < len(df) - 1:
            curr_lat = df.iloc[idx]['lat']
            curr_lon = df.iloc[idx]['lon']
            next_lat = df.iloc[idx + 1]['lat']
            next_lon = df.iloc[idx + 1]['lon']
            bearing_to_next = calculate_bearing(curr_lat, curr_lon, next_lat, next_lon)
        
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


def process_images(image_dir: Path, output_csv: Path, offset_distance_m: float = 10.0, verbose: bool = False):
    """
    Process all images in a directory and extract GPS coordinates.
    
    Args:
        image_dir: Directory containing images
        output_csv: Path to output CSV file
        offset_distance_m: Distance in meters to offset points perpendicular to heading
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
        help="Output CSV file path (format: filename, lat, lon, timestamp, heading, offset_lat, offset_lon)"
    )
    
    parser.add_argument(
        "--offset",
        type=float,
        default=-20.0,
        help="Distance in meters to offset points perpendicular to heading (default: -20.0). Positive = right side of heading."
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
    df = process_images(args.image_dir, args.output_csv, args.offset, args.verbose)
    
    # Optionally create map visualization
    if args.view_map and df is not None and not df.empty:
        visualize_on_map(args.output_csv, args.map_output)
    
    return 0


if __name__ == "__main__":
    exit(main())
