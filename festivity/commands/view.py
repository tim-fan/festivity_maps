"""
Launch FiftyOne visualization.
"""

import fiftyone as fo

from festivity.utils import get_workspace_path, ensure_workspace_initialized
from festivity.db import get_db_connection


def register_command(subparsers):
    """Register the view command."""
    parser = subparsers.add_parser(
        'view',
        help='Launch FiftyOne visualization',
        description='Explore scored images in FiftyOne'
    )
    
    parser.add_argument(
        '--workspace',
        type=str,
        help='Workspace root path (or use FESTIVITY_WORKSPACE env var)'
    )
    
    parser.set_defaults(func=execute)


def execute(args):
    """Execute the view command."""
    workspace = get_workspace_path(args)
    ensure_workspace_initialized(workspace)
    
    images_dir = workspace / 'images'
    heatmaps_dir = workspace / 'outputs' / 'heatmaps'
    
    # Get scored images from database
    conn = get_db_connection(workspace)
    cursor = conn.execute("""
        SELECT 
            s.filename,
            g.is_left,
            g.lat,
            g.lon,
            g.timestamp,
            g.heading,
            g.offset_lat,
            g.offset_lon,
            g.address,
            s.mean_probability
        FROM festivity_scores s
        LEFT JOIN gps_data g ON s.filename = g.filename
        ORDER BY s.mean_probability DESC
    """)
    
    records = cursor.fetchall()
    conn.close()
    
    if not records:
        print("No scored images found.")
        print("Run 'festivity score' first to score images.")
        return 1
    
    print(f"Loading {len(records)} scored images into FiftyOne...")
    
    # Create FiftyOne dataset
    dataset_name = f"festivity_{workspace.name}"
    
    # Delete existing dataset if it exists
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
        
    dataset = fo.Dataset(dataset_name, persistent=False)
    
    # Add samples
    samples = []
    for row in records:
        filename, is_left, lat, lon, timestamp, heading, offset_lat, offset_lon, address, mean_probability = row
        
        img_path = str(images_dir / filename)
        
        sample = fo.Sample(filepath=img_path)
        sample['filename'] = filename
        sample['festivity_score'] = float(mean_probability) if mean_probability is not None else None
        sample['is_left'] = bool(is_left)
        sample['latitude'] = float(lat) if lat is not None else None
        sample['longitude'] = float(lon) if lon is not None else None
        sample['offset_latitude'] = float(offset_lat) if offset_lat is not None else None
        sample['offset_longitude'] = float(offset_lon) if offset_lon is not None else None
        sample['timestamp'] = timestamp
        sample['heading'] = float(heading) if heading is not None else None
        sample['address'] = address
        
        # Add location field for map view
        if offset_lat is not None and offset_lon is not None:
            sample['location'] = fo.GeoLocation(point=[float(offset_lon), float(offset_lat)])
        
        # Add heatmap if it exists (FiftyOne requires absolute path)
        heatmap_filename = f"{img_path.rsplit('.', 1)[0].split('/')[-1]}_heatmap.png"
        heatmap_path = heatmaps_dir / heatmap_filename
        if heatmap_path.exists():
            sample['festivity_heatmap'] = fo.Heatmap(map_path=str(heatmap_path.absolute()))
        
        samples.append(sample)
    
    dataset.add_samples(samples)
    
    # Create index on festivity_score so it appears in the sort dropdown
    dataset.create_index("festivity_score")
    
    print(f"\nDataset: {len(dataset)} images")
    print(f"  Festivity scores: {dataset.bounds('festivity_score')}")
    
    # Sort by festivity score (highest first) by default
    view = dataset.sort_by('festivity_score', reverse=True)
    
    # Configure default sidebar fields to show festivity_score
    dataset.app_config.sidebar_mode = "all"
    dataset.app_config.sidebar_groups = [
        fo.SidebarGroupDocument(
            name="labels",
            expanded=True,
            paths=["festivity_score", "festivity_heatmap"]
        )
    ]
    dataset.save()
    
    print("\nLaunching FiftyOne App...")
    print("  - Sorted by festivity score (highest first)")
    print("  - Festivity score visible in sidebar by default")
    print("  - View 'festivity_heatmap' to see score heatmaps")
    print("  - Use Map view to see image locations")
    print("  - Press Ctrl+C to exit")
    
    session = fo.launch_app(view)
    session.wait()
