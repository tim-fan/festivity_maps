"""Histogram command - show distribution of festivity scores."""
from pathlib import Path
import sqlite3
import sys

from festivity.utils import get_workspace_path, ensure_workspace_initialized
from festivity.db import get_db_connection


def register_command(subparsers):
    """Register the histogram command."""
    parser = subparsers.add_parser(
        'histogram',
        help='Show histogram of festivity scores'
    )
    parser.add_argument(
        '--workspace',
        type=str,
        help='Path to workspace directory'
    )
    parser.add_argument(
        '--bins',
        type=int,
        default=20,
        help='Number of bins in histogram (default: 20)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=60,
        help='Width of histogram in characters (default: 60)'
    )
    parser.set_defaults(func=execute)


def execute(args):
    """Execute the histogram command."""
    workspace = get_workspace_path(args)
    ensure_workspace_initialized(workspace)
    
    # Get festivity scores with addresses from database
    conn = get_db_connection(workspace)
    cursor = conn.cursor()
    
    # Get max festivity score per address
    cursor.execute("""
        SELECT 
            g.address,
            MAX(s.mean_probability) as max_score
        FROM gps_data g
        INNER JOIN festivity_scores s ON g.filename = s.filename
        WHERE g.address IS NOT NULL AND g.address != 'none' AND g.address != ''
        GROUP BY g.address
    """)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("No festivity scores with addresses found.")
        print("Make sure you have:")
        print("  1. Extracted GPS data with addresses (festivity extract-gps)")
        print("  2. Scored images (festivity score)")
        return 1
    
    scores = [row[1] for row in rows]
    num_addresses = len(scores)
    
    # Calculate histogram
    min_score = min(scores)
    max_score = max(scores)
    bin_width = (max_score - min_score) / args.bins
    
    # Create bins
    bins = [0] * args.bins
    for score in scores:
        bin_idx = int((score - min_score) / bin_width)
        if bin_idx >= args.bins:
            bin_idx = args.bins - 1
        bins[bin_idx] += 1
    
    # Find max count for scaling
    max_count = max(bins)
    
    # Print histogram
    print("\n" + "=" * 70)
    print("FESTIVITY SCORES HISTOGRAM (Per Address)")
    print("=" * 70)
    print(f"\nTotal addresses: {num_addresses:,}")
    print(f"Score range: {min_score:.6f} - {max_score:.6f}")
    print(f"Mean score: {sum(scores)/len(scores):.6f}")
    print(f"\nDistribution ({args.bins} bins):\n")
    
    for i in range(args.bins):
        bin_start = min_score + i * bin_width
        bin_end = bin_start + bin_width
        count = bins[i]
        
        # Calculate bar width
        if max_count > 0:
            bar_width = int((count / max_count) * args.width)
        else:
            bar_width = 0
        
        # Create bar
        bar = '█' * bar_width
        
        # Print bin
        print(f"{bin_start:.6f} - {bin_end:.6f} | {bar} {count:,}")
    
    print()
    
    return 0
