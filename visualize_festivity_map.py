#!/usr/bin/env python3
"""
Visualize festivity classification on an interactive map.

This script reads GPS coordinates and inference results, classifies photos by
festivity level, and plots them on an interactive map with colored markers.
"""

import argparse
from pathlib import Path
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import shutil


def main():
    parser = argparse.ArgumentParser(
        description="Visualize festivity classification on an interactive map"
    )
    
    parser.add_argument(
        "--coords",
        type=Path,
        default=Path("coords.csv"),
        help="Path to coords.csv file (default: coords.csv)"
    )
    
    parser.add_argument(
        "--inference",
        type=Path,
        default=Path("inference_results.csv"),
        help="Path to inference_results.csv file (default: inference_results.csv)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("festivity_map.html"),
        help="Output HTML map file (default: festivity_map.html)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.008,
        help="Threshold for not_festive classification (default: 0.008)"
    )
    
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images to ./images/ folder next to HTML for image preview tooltips"
    )
    
    parser.add_argument(
        "--image-width",
        type=int,
        default=200,
        help="Width of thumbnail images in pixels (default: 200)"
    )
    
    args = parser.parse_args()
    
    # Read the CSVs
    print(f"Reading {args.coords}...")
    coords_df = pd.read_csv(args.coords)
    
    print(f"Reading {args.inference}...")
    inference_df = pd.read_csv(args.inference)
    
    # Extract just the filename from the full path in inference_results
    inference_df['filename'] = inference_df['image_path'].apply(lambda x: Path(x).name)
    
    # Merge the dataframes on filename
    print("Merging data...")
    merged_df = coords_df.merge(
        inference_df[['filename', 'mean_probability', 'image_path']], 
        on='filename', 
        how='inner'
    )
    
    print(f"Matched {len(merged_df)} out of {len(coords_df)} images from coords.csv")
    print(f"and {len(inference_df)} images from inference_results.csv")
    
    # Assert that all rows are matched
    assert len(merged_df) == len(coords_df), \
        f"Not all rows matched! {len(coords_df)} coords vs {len(merged_df)} matched"
    assert len(merged_df) == len(inference_df), \
        f"Not all rows matched! {len(inference_df)} inference vs {len(merged_df)} matched"
    
    print("✓ All rows successfully matched")
    
    # Remove rows where address is 'none'
    before_filter = len(merged_df)
    merged_df = merged_df[merged_df['address'] != 'none'].copy()
    print(f"Removed {before_filter - len(merged_df)} rows with address='none'")
    print(f"Remaining: {len(merged_df)} rows")
    
    if len(merged_df) == 0:
        print("Error: No data remaining after filtering. Exiting.")
        return 1
    
    # Classify festivity levels
    print(f"\nClassifying festivity levels (threshold={args.threshold})...")
    
    # First, separate not_festive (below threshold)
    not_festive_mask = merged_df['mean_probability'] < args.threshold
    merged_df['festivity_class'] = 'not_festive'
    
    # For the remaining photos, split into three equal groups
    festive_df = merged_df[~not_festive_mask].copy()
    
    if len(festive_df) > 0:
        # Sort by mean_probability and split into thirds
        festive_df = festive_df.sort_values('mean_probability')
        n = len(festive_df)
        
        # Calculate split points
        low_end = n // 3
        medium_end = 2 * n // 3
        
        # Assign classes
        festive_df.loc[festive_df.index[:low_end], 'festivity_class'] = 'low_festive'
        festive_df.loc[festive_df.index[low_end:medium_end], 'festivity_class'] = 'medium_festive'
        festive_df.loc[festive_df.index[medium_end:], 'festivity_class'] = 'high_festive'
        
        # Update the main dataframe
        merged_df.loc[festive_df.index, 'festivity_class'] = festive_df['festivity_class']
    
    # Print classification summary
    print("\nFestivity classification summary:")
    for cls in ['not_festive', 'low_festive', 'medium_festive', 'high_festive']:
        count = (merged_df['festivity_class'] == cls).sum()
        if count > 0:
            probs = merged_df[merged_df['festivity_class'] == cls]['mean_probability']
            print(f"  {cls}: {count} photos (probability range: {probs.min():.6f} - {probs.max():.6f})")
        else:
            print(f"  {cls}: 0 photos")
    
    # Copy images if requested
    images_dir = None
    if args.copy_images:
        print("\nCopying images...")
        images_dir = args.output.parent / "images"
        images_dir.mkdir(exist_ok=True)
        
        copied = 0
        for img_path in merged_df['image_path'].unique():
            src = Path(img_path)
            if src.exists():
                dst = images_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
                    copied += 1
        
        print(f"  Copied {copied} images to {images_dir}")

    
    # Create the map
    print("\nCreating map...")
    
    # Use address lat/lon for marker positions
    center_lat = merged_df['address_lat'].mean()
    center_lon = merged_df['address_lon'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    
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
    address_groups = merged_df.groupby(['address', 'address_lat', 'address_lon'])
    
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
        
        # Create tooltip with image (if images were copied)
        if images_dir:
            img_filename = Path(best_image_row['image_path']).name
            tooltip_html = f"""
            <div style="text-align: center;">
                <img src="images/{img_filename}" width="{args.image_width}px"><br>
                <b>{address}</b><br>
                {photo_count} photo{"s" if photo_count > 1 else ""} - {dominant_class.replace('_', ' ').title()}
            </div>
            """
            tooltip = folium.Tooltip(tooltip_html)
        else:
            tooltip = f"{address} ({photo_count} photos)"
        
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
            popup_html += f"<li>{row['filename']}: {row['mean_probability']:.4f} ({row['festivity_class']})</li>"
        
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
    m.save(str(args.output))
    print(f"\n✓ Map saved to {args.output}")
    print(f"  Open this file in a web browser to view the interactive map")
    
    return 0


if __name__ == "__main__":
    exit(main())
