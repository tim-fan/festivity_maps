"""Map command - visualize festivity scores on an interactive map."""
from pathlib import Path
import pandas as pd
import numpy as np
import folium
from folium.plugins import LocateControl
import shutil
from PIL import Image
from scipy.spatial import cKDTree
from festivity.utils import get_workspace_path, ensure_workspace_initialized
from festivity.db import get_db_connection
from festivity.trajectory import calculate_trajectory_stats, format_duration, format_distance, format_speed


def register_command(subparsers):
    """Register the map command."""
    parser = subparsers.add_parser(
        'map',
        help='Generate interactive map visualization of festivity scores'
    )
    parser.add_argument(
        '--workspace',
        type=str,
        help='Path to workspace directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output HTML file path (default: workspace/outputs/festivity_map.html)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help='Threshold for lights detected classification (default: 0.01)'
    )
    parser.add_argument(
        '--image-width',
        type=int,
        default=400,
        help='Width of thumbnail images in pixels (default: 400)'
    )
    parser.add_argument(
        '--show-addresses',
        action='store_true',
        help='Show street addresses on map and in tooltips (default: hidden)'
    )
    parser.add_argument(
        '--deploy',
        type=str,
        metavar='DIR',
        help='Create a self-contained deployment directory with HTML and images (for uploading to Netlify, etc.)'
    )
    parser.set_defaults(func=execute)


def cluster_nearby_addresses(df, distance_threshold_m=10):
    """
    Cluster addresses that are within distance_threshold_m of each other.
    Returns a dataframe with an additional 'cluster_id' column.
    
    Uses spatial clustering to group nearby markers, preventing overlapping
    markers from OSM addresses that are very close together.
    """
    # Get unique address locations
    unique_locs = df[['address_lat', 'address_lon']].drop_duplicates()
    
    if len(unique_locs) == 0:
        df['cluster_id'] = 0
        return df
    
    # Build KD-tree for clustering
    coords = unique_locs[['address_lat', 'address_lon']].values
    tree = cKDTree(coords)
    
    # Convert distance threshold from meters to degrees (approximate)
    # At mid-latitudes: 1 degree ≈ 111km
    distance_threshold_deg = distance_threshold_m / 111000.0
    
    # Find all pairs within threshold and build clusters
    pairs = tree.query_pairs(r=distance_threshold_deg)
    
    # Use union-find to merge clusters
    cluster_map = {}
    next_cluster_id = 0
    
    def find_cluster(idx):
        """Find root cluster for an index."""
        if idx not in cluster_map:
            return None
        root = idx
        while cluster_map[root] != root:
            root = cluster_map[root]
        return root
    
    def merge_clusters(idx1, idx2):
        """Merge two indices into the same cluster."""
        nonlocal next_cluster_id
        c1 = find_cluster(idx1)
        c2 = find_cluster(idx2)
        
        if c1 is None and c2 is None:
            # Both new - create new cluster
            cluster_map[idx1] = idx1
            cluster_map[idx2] = idx1
        elif c1 is None:
            # idx1 is new, add to idx2's cluster
            cluster_map[idx1] = c2
        elif c2 is None:
            # idx2 is new, add to idx1's cluster
            cluster_map[idx2] = c1
        elif c1 != c2:
            # Merge two existing clusters
            cluster_map[c2] = c1
    
    # Process all pairs
    for idx1, idx2 in pairs:
        merge_clusters(idx1, idx2)
    
    # Assign cluster IDs to all points
    idx_to_cluster = {}
    for idx in range(len(coords)):
        root = find_cluster(idx)
        if root is None:
            # Singleton cluster
            idx_to_cluster[idx] = next_cluster_id
            next_cluster_id += 1
        else:
            if root not in idx_to_cluster:
                idx_to_cluster[root] = next_cluster_id
                next_cluster_id += 1
            idx_to_cluster[idx] = idx_to_cluster[root]
    
    # Map back to original dataframe
    unique_locs['temp_idx'] = range(len(unique_locs))
    unique_locs['cluster_id'] = unique_locs['temp_idx'].map(idx_to_cluster)
    
    # Merge cluster_id back into original dataframe
    df = df.merge(
        unique_locs[['address_lat', 'address_lon', 'cluster_id']], 
        on=['address_lat', 'address_lon'], 
        how='left'
    )
    
    return df


def execute(args):
    """Execute the map command."""
    workspace = get_workspace_path(args)
    ensure_workspace_initialized(workspace)
    
    # Handle deployment mode
    deploy_dir = None
    if args.deploy:
        deploy_dir = Path(args.deploy).expanduser().resolve()
        
        # Check if directory exists and is non-empty
        if deploy_dir.exists():
            if any(deploy_dir.iterdir()):
                print(f"Error: Deploy directory '{deploy_dir}' exists and is not empty.")
                print("Please specify an empty directory or delete the existing contents.")
                return 1
        else:
            # Create the directory
            deploy_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created deploy directory: {deploy_dir}")
        
        # Set output to deploy directory
        output_path = deploy_dir / 'festivity_map.html'
    elif args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        outputs_dir = workspace / 'outputs'
        outputs_dir.mkdir(exist_ok=True)
        output_path = outputs_dir / 'festivity_map.html'
    
    # Get data from database
    print("Loading data from database...")
    conn = get_db_connection(workspace)
    
    # Query for images with GPS and festivity scores
    query = """
    SELECT 
        g.filename,
        g.is_left,
        g.lat,
        g.lon,
        g.heading,
        g.offset_lat,
        g.offset_lon,
        g.address,
        g.address_lat,
        g.address_lon,
        g.timestamp,
        s.mean_probability
    FROM gps_data g
    INNER JOIN festivity_scores s ON g.filename = s.filename
    WHERE g.address IS NOT NULL AND g.address != 'none'
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) == 0:
        print("Error: No data found. Make sure you have:")
        print("  1. Extracted GPS data (festivity extract-gps)")
        print("  2. Scored images (festivity score)")
        print("  3. Fetched addresses (extract-gps without --skip-address-fetch)")
        return 1
    
    print(f"Loaded {len(df)} images with GPS and festivity scores")
    
    # Classify festivity levels
    print(f"\nClassifying festivity levels (threshold={args.threshold})...")
    
    # First, separate no_lights (below threshold)
    no_lights_mask = df['mean_probability'] < args.threshold
    df['festivity_class'] = 'no_lights'
    
    # For images with lights detected (>= threshold), use percentile-based classification
    festive_df = df[~no_lights_mask].copy()
    
    if len(festive_df) > 0:
        # Sort by mean_probability
        festive_df = festive_df.sort_values('mean_probability')
        n = len(festive_df)
        
        # Calculate percentile cutoffs
        # Top 1% -> most_lights (yellow)
        # Top 10% (excluding top 1%) -> many_lights (orange)
        # Rest (bottom 90%) -> lights_detected (red)
        top_1_percent_idx = int(n * 0.99)  # Start of top 1%
        top_10_percent_idx = int(n * 0.90)  # Start of top 10%
        
        # Assign classes based on percentiles
        bottom_indices = festive_df.index[:top_10_percent_idx]
        middle_indices = festive_df.index[top_10_percent_idx:top_1_percent_idx]
        top_indices = festive_df.index[top_1_percent_idx:]
        
        df.loc[bottom_indices, 'festivity_class'] = 'lights_detected'
        df.loc[middle_indices, 'festivity_class'] = 'many_lights'
        df.loc[top_indices, 'festivity_class'] = 'most_lights'
    
    # Print classification summary
    print("\nFestivity classification summary:")
    for cls in ['no_lights', 'lights_detected', 'many_lights', 'most_lights']:
        count = (df['festivity_class'] == cls).sum()
        if count > 0:
            probs = df[df['festivity_class'] == cls]['mean_probability']
            print(f"  {cls.replace('_', ' ')}: {count} photos (probability range: {probs.min():.6f} - {probs.max():.6f})")
        else:
            print(f"  {cls.replace('_', ' ')}: 0 photos")
    
    # Create the map
    print("\nCreating map...")
    
    # Use address lat/lon for centering
    # Create the map centered on data
    center_lat = df['address_lat'].mean()
    center_lon = df['address_lon'].mean()
    
    # Use minimal style when addresses are hidden, default OSM when showing addresses
    if args.show_addresses:
        m = folium.Map(location=[center_lat, center_lon])
    else:
        # Use CartoDB Positron for a minimal style without street labels
        m = folium.Map(
            location=[center_lat, center_lon],
            tiles='CartoDB positron',
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        )
    
    # Calculate bounds for zoom-to-extents
    sw = [df['address_lat'].min(), df['address_lon'].min()]
    ne = [df['address_lat'].max(), df['address_lon'].max()]
    m.fit_bounds([sw, ne], padding=[30, 30])  # 30px padding
    
    # Add user location control (shows GPS location)
    LocateControl(auto_start=False).add_to(m)
    
    # Add camera trajectory tracks (left and right)
    print("Adding camera trajectories...")
    
    # Get all GPS data for trajectories (not just scored images)
    conn = get_db_connection(workspace)
    trajectory_query = """
    SELECT filename, is_left, lat, lon, offset_lat, offset_lon, timestamp
    FROM gps_data
    WHERE lat IS NOT NULL AND lon IS NOT NULL
    ORDER BY timestamp
    """
    trajectory_df = pd.read_sql_query(trajectory_query, conn)
    conn.close()
    
    # Create feature groups for left and right camera tracks (actual GPS)
    left_track_group = folium.FeatureGroup(name='Left Camera Track (GPS)', show=False)
    right_track_group = folium.FeatureGroup(name='Right Camera Track (GPS)', show=False)
    
    # Create feature groups for offset tracks (perpendicular offset from GPS)
    left_offset_group = folium.FeatureGroup(name='Left Camera Track (Offset)', show=False)
    right_offset_group = folium.FeatureGroup(name='Right Camera Track (Offset)', show=False)
    
    # Separate left and right tracks
    left_coords = trajectory_df[trajectory_df['is_left'] == True][['lat', 'lon']].values.tolist()
    right_coords = trajectory_df[trajectory_df['is_left'] == False][['lat', 'lon']].values.tolist()
    
    left_offset_coords = trajectory_df[trajectory_df['is_left'] == True][['offset_lat', 'offset_lon']].values.tolist()
    right_offset_coords = trajectory_df[trajectory_df['is_left'] == False][['offset_lat', 'offset_lon']].values.tolist()
    
    # Add polylines for GPS tracks
    if len(left_coords) > 0:
        folium.PolyLine(
            left_coords,
            color='blue',
            weight=2,
            opacity=0.6,
            popup='Left Camera GPS Track'
        ).add_to(left_track_group)
    
    if len(right_coords) > 0:
        folium.PolyLine(
            right_coords,
            color='green',
            weight=2,
            opacity=0.6,
            popup='Right Camera GPS Track'
        ).add_to(right_track_group)
    
    # Add polylines for offset tracks
    if len(left_offset_coords) > 0:
        folium.PolyLine(
            left_offset_coords,
            color='lightblue',
            weight=2,
            opacity=0.6,
            dash_array='5, 5',
            popup='Left Camera Offset Track'
        ).add_to(left_offset_group)
    
    if len(right_offset_coords) > 0:
        folium.PolyLine(
            right_offset_coords,
            color='lightgreen',
            weight=2,
            opacity=0.6,
            dash_array='5, 5',
            popup='Right Camera Offset Track'
        ).add_to(right_offset_group)
    
    # Add track groups to map
    left_track_group.add_to(m)
    right_track_group.add_to(m)
    left_offset_group.add_to(m)
    right_offset_group.add_to(m)
    
    # Define colors for each festivity class
    color_map = {
        'no_lights': 'gray',
        'lights_detected': 'darkred',
        'many_lights': 'orange',
        'most_lights': 'yellow'
    }
    
    # Display names for categories
    display_names = {
        'no_lights': 'No Lights Detected',
        'lights_detected': 'Lights Detected',
        'many_lights': 'Many Lights Detected',
        'most_lights': 'Most Lights Detected'
    }
    
    # Create feature groups for each festivity class
    feature_groups = {}
    for cls in ['no_lights', 'lights_detected', 'many_lights', 'most_lights']:
        show = cls != 'no_lights'  # Hide no_lights by default
        feature_groups[cls] = folium.FeatureGroup(name=display_names[cls], show=show)
    
    # Cluster nearby addresses to avoid overlapping markers
    print("Clustering nearby addresses...")
    df = cluster_nearby_addresses(df, distance_threshold_m=10)
    
    # Group by cluster instead of individual addresses
    print("Grouping photos by location cluster...")
    cluster_groups = df.groupby('cluster_id')
    
    # Track images used in tooltips for deployment
    used_images = set()
    
    for cluster_id, group in cluster_groups:
        # Use the centroid of all addresses in this cluster
        lat = group['address_lat'].mean()
        lon = group['address_lon'].mean()
        
        # Collect unique addresses in this cluster
        unique_addresses = group['address'].unique()
        address_display = unique_addresses[0] if len(unique_addresses) == 1 else f"{len(unique_addresses)} addresses"
        
        # Determine the highest festivity class in this cluster
        class_priority = {'most_lights': 4, 'many_lights': 3, 'lights_detected': 2, 'no_lights': 1}
        dominant_class = group.loc[group['festivity_class'].map(class_priority).idxmax(), 'festivity_class']
        
        # Get the highest probability image for the tooltip
        best_image_row = group.loc[group['mean_probability'].idxmax()]
        
        # Get stats for this cluster
        photo_count = len(group)
        mean_prob = group['mean_probability'].mean()
        max_prob = group['mean_probability'].max()
        
        # Format timestamp for display
        timestamp_str = ""
        if pd.notna(best_image_row.get('timestamp')):
            try:
                from datetime import datetime
                ts = pd.to_datetime(best_image_row['timestamp'])
                timestamp_str = f"<br>{ts.strftime('%Y-%m-%d %H:%M:%S')}"
            except:
                pass
        
        # Create tooltip with image
        img_filename = best_image_row['filename']
        used_images.add(img_filename)  # Track this image for deployment
        
        # Image path depends on deployment mode
        if deploy_dir:
            img_path = f"images/{img_filename}"
        else:
            img_path = f"../images/{img_filename}"
        
        if args.show_addresses:
            tooltip_html = f"""
            <div style="text-align: center; background-color: white; padding: 5px;">
                <img src="{img_path}" width="{args.image_width}px" style="opacity: 1.0;"><br>
                <b>{address_display}</b>{timestamp_str}<br>
                {photo_count} photo{"s" if photo_count > 1 else ""} - {display_names[dominant_class]}<br>
                Avg: {mean_prob:.4f} | Max: {max_prob:.4f}
            </div>
            """
        else:
            tooltip_html = f"""
            <div style="text-align: center; background-color: white; padding: 5px;">
                <img src="{img_path}" width="{args.image_width}px" style="opacity: 1.0;">{timestamp_str}<br>
                {photo_count} photo{"s" if photo_count > 1 else ""} - {display_names[dominant_class]}<br>
                Avg: {mean_prob:.4f} | Max: {max_prob:.4f}
            </div>
            """
        tooltip = folium.Tooltip(tooltip_html, sticky=False)
        
        # Add marker to the appropriate feature group
        # Use Circle (radius in meters) instead of CircleMarker (radius in pixels)
        # so markers scale with zoom level
        # Make "most lights" markers bigger for emphasis
        radius_map = {
            'no_lights': 7,
            'lights_detected': 8,
            'many_lights': 10,
            'most_lights': 15
        }
        
        # Style "most lights" markers differently to stand out
        if dominant_class == 'most_lights':
            # Darker border and thicker outline for yellow markers
            border_color = "#72744B"  # 
            border_weight = 1
        else:
            # Standard styling for other classes
            border_color = color_map[dominant_class]
            border_weight = 1
        
        folium.Circle(
            location=[lat, lon],
            radius=radius_map[dominant_class],  # meters
            tooltip=tooltip,
            color=border_color,
            fill=True,
            fillColor=color_map[dominant_class],
            fillOpacity=0.7,
            weight=border_weight
        ).add_to(feature_groups[dominant_class])
    
    # Add all feature groups to map
    for fg in feature_groups.values():
        fg.add_to(m)
    
    # Add layer control (collapsed by default)
    folium.LayerControl(collapsed=True).add_to(m)
    
    # Calculate trajectory statistics
    print("Calculating trajectory statistics...")
    db_path = workspace / 'database.db'
    traj_stats = calculate_trajectory_stats(db_path)
    
    # Calculate average speed if duration > 0
    avg_speed_html = ""
    if traj_stats['total_duration_seconds'] > 0:
        avg_speed = traj_stats['total_distance_m'] / traj_stats['total_duration_seconds']
        avg_speed_html = f"<div style='margin-top: 3px;'><b>{format_speed(avg_speed)}</b> avg speed</div>"
    
    # Add trajectory statistics as a collapsible control box in lower left
    stats_html = f'''
    <div id="trajectory-stats-control" style="position: fixed; 
                bottom: 30px; left: 10px; 
                background-color: white; 
                border: 2px solid rgba(0,0,0,0.2);
                border-radius: 4px;
                font-family: Arial, sans-serif;
                font-size: 12px;
                z-index: 1000;
                box-shadow: 0 1px 5px rgba(0,0,0,0.4);
                cursor: pointer;">
        <div id="stats-header" style="padding: 6px 10px; 
                    background-color: white; 
                    border-radius: 4px;
                    font-weight: bold;">
            <span id="stats-toggle">▶</span> Trajectory Stats
        </div>
        <div id="stats-content" style="display: none; 
                    padding: 8px 10px; 
                    border-top: 1px solid rgba(0,0,0,0.2);">
            <div style='margin-top: 3px;'><b>{traj_stats['total_images']:,}</b> images</div>
            <div style='margin-top: 3px;'><b>{format_distance(traj_stats['total_distance_m'])}</b> distance</div>
            <div style='margin-top: 3px;'><b>{format_duration(traj_stats['total_duration_seconds'])}</b> duration</div>
            {avg_speed_html}
            <div style='margin-top: 3px;'><b>{traj_stats['unique_addresses']:,}</b> unique addresses</div>
            <div style='margin-top: 8px; padding-top: 5px; border-top: 1px solid rgba(0,0,0,0.1); font-size: 10px;'>
                <a href="https://github.com/tim-fan/festivity_maps" target="_blank" style="color: #0066cc; text-decoration: none;">
                    🔗 github.com/tim-fan/festivity_maps
                </a>
            </div>
        </div>
    </div>
    <script>
        document.getElementById('trajectory-stats-control').addEventListener('click', function(e) {{
            var content = document.getElementById('stats-content');
            var toggle = document.getElementById('stats-toggle');
            if (content.style.display === 'none') {{
                content.style.display = 'block';
                toggle.textContent = '▼';
            }} else {{
                content.style.display = 'none';
                toggle.textContent = '▶';
            }}
        }});
    </script>
    '''
    
    m.get_root().html.add_child(folium.Element(stats_html))
    
    # Add custom CSS to override Folium's default tooltip opacity
    tooltip_css = '''
    <style>
        .leaflet-tooltip {
            opacity: 1.0 !important;
            background-color: white !important;
            border: 2px solid gray !important;
        }
        .leaflet-tooltip img {
            opacity: 1.0 !important;
        }
    </style>
    '''
    m.get_root().html.add_child(folium.Element(tooltip_css))
    
    # Save the map
    m.save(str(output_path))
    print(f"\n✓ Map saved to {output_path}")
    
    # Copy images for deployment
    if deploy_dir:
        images_dir = deploy_dir / 'images'
        images_dir.mkdir(exist_ok=True)
        
        print(f"\nResizing and copying {len(used_images)} images to deployment directory...")
        workspace_images_dir = workspace / 'images'
        
        # Resize images to save bandwidth (max width based on tooltip setting)
        max_width = args.image_width * 2  # 2x for retina displays
        
        for img_filename in used_images:
            src = workspace_images_dir / img_filename
            dst = images_dir / img_filename
            if src.exists():
                try:
                    # Open and resize image
                    with Image.open(src) as img:
                        # Calculate new size maintaining aspect ratio
                        if img.width > max_width:
                            ratio = max_width / img.width
                            new_size = (max_width, int(img.height * ratio))
                            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
                            # Save with optimization
                            img_resized.save(dst, quality=85, optimize=True)
                        else:
                            # Image is already small enough, just copy
                            shutil.copy2(src, dst)
                except Exception as e:
                    print(f"Warning: Failed to resize {img_filename}: {e}")
                    # Fall back to direct copy
                    shutil.copy2(src, dst)
            else:
                print(f"Warning: Image not found: {src}")
        
        print(f"\n✓ Deployment package created in {deploy_dir}")
        print(f"  - festivity_map.html")
        print(f"  - images/ ({len(used_images)} files, resized to max {max_width}px width)")
        print(f"\n  You can now upload this directory to Netlify or any static hosting service.")
    else:
        print(f"\n  To view the map with image tooltips, start a web server:")
        print(f"    cd {workspace}")
        print(f"    python3 -m http.server 8000")
        print(f"  Then open: http://localhost:8000/outputs/{output_path.name}")
    
    return 0
