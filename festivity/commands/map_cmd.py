"""Map command - visualize festivity scores on an interactive map."""
from pathlib import Path
import pandas as pd
import folium
from festivity.utils import get_workspace_path, ensure_workspace_initialized
from festivity.db import get_db_connection


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
    parser.set_defaults(func=execute)


def execute(args):
    """Execute the map command."""
    workspace = get_workspace_path(args)
    ensure_workspace_initialized(workspace)
    
    # Set output path
    if args.output:
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
    
    # First, separate not_festive (below threshold)
    not_festive_mask = df['mean_probability'] < args.threshold
    df['festivity_class'] = 'not_festive'
    
    # For the remaining photos, split into three equal groups
    festive_df = df[~not_festive_mask].copy()
    
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
        
        df.loc[low_indices, 'festivity_class'] = 'low_festive'
        df.loc[medium_indices, 'festivity_class'] = 'medium_festive'
        df.loc[high_indices, 'festivity_class'] = 'high_festive'
    
    # Print classification summary
    print("\nFestivity classification summary:")
    for cls in ['not_festive', 'low_festive', 'medium_festive', 'high_festive']:
        count = (df['festivity_class'] == cls).sum()
        if count > 0:
            probs = df[df['festivity_class'] == cls]['mean_probability']
            print(f"  {cls}: {count} photos (probability range: {probs.min():.6f} - {probs.max():.6f})")
        else:
            print(f"  {cls}: 0 photos")
    
    # Create the map
    print("\nCreating map...")
    
    # Use address lat/lon for centering
    center_lat = df['address_lat'].mean()
    center_lon = df['address_lon'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    
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
        'not_festive': 'gray',
        'low_festive': 'darkred',
        'medium_festive': 'orange',
        'high_festive': 'yellow'
    }
    
    # Create feature groups for each festivity class
    feature_groups = {}
    for cls in ['not_festive', 'low_festive', 'medium_festive', 'high_festive']:
        show = cls != 'not_festive'  # Hide not_festive by default
        feature_groups[cls] = folium.FeatureGroup(name=cls.replace('_', ' ').title(), show=show)
    
    # Group by address to avoid duplicate markers
    print("Grouping photos by address...")
    address_groups = df.groupby(['address', 'address_lat', 'address_lon'])
    
    for (address, lat, lon), group in address_groups:
        # Determine the highest festivity class at this address
        class_priority = {'high_festive': 4, 'medium_festive': 3, 'low_festive': 2, 'not_festive': 1}
        dominant_class = group.loc[group['festivity_class'].map(class_priority).idxmax(), 'festivity_class']
        
        # Get the highest probability image for the tooltip
        best_image_row = group.loc[group['mean_probability'].idxmax()]
        
        # Get stats for this address
        photo_count = len(group)
        mean_prob = group['mean_probability'].mean()
        max_prob = group['mean_probability'].max()
        
        # Create tooltip with image
        img_filename = best_image_row['filename']
        tooltip_html = f"""
        <div style="text-align: center;">
            <img src="../images/{img_filename}" width="{args.image_width}px"><br>
            <b>{address}</b><br>
            {photo_count} photo{"s" if photo_count > 1 else ""} - {dominant_class.replace('_', ' ').title()}
        </div>
        """
        tooltip = folium.Tooltip(tooltip_html)
        
        # Create popup text
        popup_html = f"""
        <b>{address}</b><br>
        <br>
        Class: <b>{dominant_class.replace('_', ' ').title()}</b><br>
        Photos: {photo_count}<br>
        Avg probability: {mean_prob:.4f}<br>
        Max probability: {max_prob:.4f}<br>
        <br>
        <details>
        <summary>Photos ({photo_count})</summary>
        <ul style="margin: 5px 0; padding-left: 20px;">
        """
        
        for _, row in group.iterrows():
            side = "left" if row['is_left'] else "right"
            popup_html += f"<li>{row['filename']} ({side}): {row['mean_probability']:.4f} ({row['festivity_class']})</li>"
        
        popup_html += """
        </ul>
        </details>
        """
        
        # Add marker to the appropriate feature group
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            popup=folium.Popup(popup_html, max_width=300),
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
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p style="margin: 0 0 10px 0; font-weight: bold;">Festivity Classification</p>
    <p style="margin: 5px 0;"><i class="fa fa-circle" style="color:gray"></i> Not Festive (< {threshold})</p>
    <p style="margin: 5px 0;"><i class="fa fa-circle" style="color:darkred"></i> Low Festive</p>
    <p style="margin: 5px 0;"><i class="fa fa-circle" style="color:orange"></i> Medium Festive</p>
    <p style="margin: 5px 0;"><i class="fa fa-circle" style="color:yellow"></i> High Festive</p>
    </div>
    '''.format(threshold=args.threshold)
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save the map
    m.save(str(output_path))
    print(f"\n✓ Map saved to {output_path}")
    print(f"\n  To view the map with image tooltips, start a web server:")
    print(f"    cd {workspace}")
    print(f"    python3 -m http.server 8000")
    print(f"  Then open: http://localhost:8000/outputs/{output_path.name}")
    
    return 0
