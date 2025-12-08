"""Trajectory statistics command - show trajectory statistics."""
from pathlib import Path
from festivity.utils import get_workspace_path, ensure_workspace_initialized
from festivity.trajectory import calculate_trajectory_stats, format_duration, format_distance, format_speed


def register_command(subparsers):
    """Register the trajectory-stats command."""
    parser = subparsers.add_parser(
        'trajectory-stats',
        help='Show trajectory statistics (distance, duration, etc.)'
    )
    parser.add_argument(
        '--workspace',
        type=str,
        help='Path to workspace directory'
    )
    parser.add_argument(
        '--gap-threshold',
        type=float,
        default=1800.0,
        help='Maximum gap in seconds before treating as new sub-trajectory (default: 1800.0 / 30 minutes)'
    )
    parser.add_argument(
        '--min-distance',
        type=float,
        default=2.0,
        help='Minimum distance in meters to count toward trajectory length (default: 2.0)'
    )
    parser.set_defaults(func=execute)


def execute(args):
    """Execute the trajectory-stats command."""
    workspace = get_workspace_path(args)
    ensure_workspace_initialized(workspace)
    
    db_path = workspace / 'database.db'
    
    print("Calculating trajectory statistics...")
    stats = calculate_trajectory_stats(
        db_path, 
        gap_threshold_seconds=args.gap_threshold,
        min_distance_m=args.min_distance
    )
    
    if stats['total_images'] == 0:
        print("\nNo GPS data found.")
        print("Run 'festivity extract-gps' to extract GPS data from images.")
        return 1
    
    print("\n" + "=" * 60)
    print("TRAJECTORY STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal images:        {stats['total_images']:,}")
    print(f"Total distance:      {format_distance(stats['total_distance_m'])}")
    print(f"Total duration:      {format_duration(stats['total_duration_seconds'])}")
    
    # Calculate and display average speed
    if stats['total_duration_seconds'] > 0:
        avg_speed = stats['total_distance_m'] / stats['total_duration_seconds']
        print(f"Average speed:       {format_speed(avg_speed)}")
    
    print(f"Unique addresses:    {stats['unique_addresses']:,}")
    
    if stats['num_sub_trajectories'] > 1:
        print(f"Sub-trajectories:    {stats['num_sub_trajectories']}")
    
    # Show distance filtering info
    print(f"\n(Distance filter: {args.min_distance}m, Gap threshold: {args.gap_threshold}s)")
    
    if stats['num_sub_trajectories'] > 1:
        print("\nSub-trajectory details:")
        print("-" * 60)
        
        for i, sub in enumerate(stats['sub_trajectories'], 1):
            print(f"\nSub-trajectory {i}:")
            print(f"  Images:    {sub['images']}")
            print(f"  Distance:  {format_distance(sub['distance_m'])}")
            print(f"  Duration:  {format_duration(sub['duration_seconds'])}")
            print(f"  Start:     {sub['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  End:       {sub['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            if sub['duration_seconds'] > 0:
                avg_speed = sub['distance_m'] / sub['duration_seconds']
                print(f"  Avg speed: {format_speed(avg_speed)}")
    
    print("\n" + "=" * 60)
    
    return 0

