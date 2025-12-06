"""Reset/wipe database tables."""
import sqlite3
import shutil
from festivity.utils import get_workspace_path, ensure_workspace_initialized
from festivity.db import get_db_connection


def register_command(subparsers):
    """Register the reset command."""
    parser = subparsers.add_parser(
        'reset',
        help='Reset database (clear all or specific tables)',
        description='Clear database tables. Use with caution!'
    )
    parser.add_argument(
        '--workspace',
        type=str,
        help='Path to workspace directory'
    )
    parser.add_argument(
        '--scores',
        action='store_true',
        help='Clear festivity_scores table only'
    )
    parser.add_argument(
        '--gps',
        action='store_true',
        help='Clear gps_data table only'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Clear all tables (default if no specific table specified)'
    )
    parser.add_argument(
        '--yes',
        action='store_true',
        help='Skip confirmation prompt'
    )
    parser.set_defaults(func=execute)


def execute(args):
    """Execute the reset command."""
    workspace = get_workspace_path(args)
    ensure_workspace_initialized(workspace)
    
    # Determine which tables to clear
    # Default: clear all tables
    if args.scores or args.gps:
        clear_scores = args.scores
        clear_gps = args.gps
    else:
        # Default: clear all tables
        clear_scores = True
        clear_gps = True
    
    # Build message
    tables_to_clear = []
    if clear_scores:
        tables_to_clear.append("festivity_scores")
    if clear_gps:
        tables_to_clear.append("gps_data")
    
    if not tables_to_clear:
        print("No tables to clear.")
        return 0
    
    # Confirmation prompt
    if not args.yes:
        print(f"\nWarning: This will DELETE ALL DATA from:")
        for table in tables_to_clear:
            print(f"  - {table} (database table)")
        print(f"  - outputs/ (heatmaps and other generated files)")
        print(f"\nWorkspace: {workspace}")
        response = input("\nAre you sure? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled.")
            return 0
    
    # Clear database tables
    conn = get_db_connection(workspace)
    cursor = conn.cursor()
    
    for table in tables_to_clear:
        print(f"Clearing {table}...")
        cursor.execute(f"DELETE FROM {table}")
        rows_deleted = cursor.rowcount
        print(f"  Deleted {rows_deleted} rows")
    
    conn.commit()
    conn.close()
    
    # Clear outputs directory contents (but keep the directory)
    outputs_dir = workspace / 'outputs'
    if outputs_dir.exists():
        print(f"Clearing outputs directory...")
        files_deleted = 0
        for item in outputs_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            files_deleted += 1
        print(f"  Deleted {files_deleted} items from {outputs_dir}")
    
    print("\nReset complete.")
    return 0
