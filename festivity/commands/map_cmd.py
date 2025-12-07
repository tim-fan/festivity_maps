"""Map command - visualize festivity scores on an interactive map."""
from pathlib import Path
import pandas as pd
import folium
from folium.plugins import LocateControl
import shutil
from PIL import Image
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
        default=0.008,
        help='Threshold for not_festive classification (default: 0.008)'
    )
    parser.add_argument(
        '--image-width',
        type=int,
        default=200,
        help='Width of thumbnail images in pixels (default: 200)'
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
    
    # For the remaining photos, split into three equal groups
    festive_df = df[~no_lights_mask].copy()
    
    if len(festive_df) > 0:
        # Sort by mean_probability and split into thirds
        festive_df = festive_df.sort_values('mean_probability')
        n = len(festive_df)
        
        # Calculate split points
        low_end = n // 3
        medium_end = 2 * n // 3
        
        # Assign classes based on index position
        low_indices = festive_df.index[:low_end]
        medium_indices = festive_df.index[low_end:medium_end]
        high_indices = festive_df.index[medium_end:]
        
        df.loc[low_indices, 'festivity_class'] = 'some_lights'
        df.loc[medium_indices, 'festivity_class'] = 'moderate_lights'
        df.loc[high_indices, 'festivity_class'] = 'many_lights'
    
    # Print classification summary
    print("\nFestivity classification summary:")
    for cls in ['no_lights', 'some_lights', 'moderate_lights', 'many_lights']:
        count = (df['festivity_class'] == cls).sum()
        if count > 0:
            probs = df[df['festivity_class'] == cls]['mean_probability']
            print(f"  {cls.replace('_', ' ')}: {count} photos (probability range: {probs.min():.6f} - {probs.max():.6f})")
        else:
            print(f"  {cls.replace('_', ' ')}: 0 photos")
        if count > 0:
            probs = df[df['festivity_class'] == cls]['mean_probability']
            print(f"  {cls}: {count} photos (probability range: {probs.min():.6f} - {probs.max():.6f})")
        else:
            print(f"  {cls}: 0 photos")
    
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
        'some_lights': 'darkred',
        'moderate_lights': 'orange',
        'many_lights': 'yellow'
    }
    
    # Display names for categories
    display_names = {
        'no_lights': 'No Lights Detected',
        'some_lights': 'Some Lights Detected',
        'moderate_lights': 'Moderate Lights Detected',
        'many_lights': 'Many Lights Detected'
    }
    
    # Create feature groups for each festivity class
    feature_groups = {}
    for cls in ['no_lights', 'some_lights', 'moderate_lights', 'many_lights']:
        show = cls != 'no_lights'  # Hide no_lights by default
        feature_groups[cls] = folium.FeatureGroup(name=display_names[cls], show=show)
    
    # Group by address to avoid duplicate markers
    print("Grouping photos by address...")
    address_groups = df.groupby(['address', 'address_lat', 'address_lon'])
    
    # Track images used in tooltips for deployment
    used_images = set()
    
    for (address, lat, lon), group in address_groups:
        # Determine the highest festivity class at this address
        class_priority = {'many_lights': 4, 'moderate_lights': 3, 'some_lights': 2, 'no_lights': 1}
        dominant_class = group.loc[group['festivity_class'].map(class_priority).idxmax(), 'festivity_class']
        
        # Get the highest probability image for the tooltip
        best_image_row = group.loc[group['mean_probability'].idxmax()]
        
        # Get stats for this address
        photo_count = len(group)
        mean_prob = group['mean_probability'].mean()
        max_prob = group['mean_probability'].max()
        
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
            <div style="text-align: center;">
                <img src="{img_path}" width="{args.image_width}px"><br>
                <b>{address}</b><br>
                {photo_count} photo{"s" if photo_count > 1 else ""} - {display_names[dominant_class]}<br>
                Avg: {mean_prob:.4f} | Max: {max_prob:.4f}
            </div>
            """
        else:
            tooltip_html = f"""
            <div style="text-align: center;">
                <img src="{img_path}" width="{args.image_width}px"><br>
                {photo_count} photo{"s" if photo_count > 1 else ""} - {display_names[dominant_class]}<br>
                Avg: {mean_prob:.4f} | Max: {max_prob:.4f}
            </div>
            """
        tooltip = folium.Tooltip(tooltip_html)
        
        # Add marker to the appropriate feature group
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            tooltip=tooltip,
            color=color_map[dominant_class],
            fill=True,
            fillColor=color_map[dominant_class],
            fillOpacity=0.7,
            weight=2
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
