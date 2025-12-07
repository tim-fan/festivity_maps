"""
Initialize festivity workspace.
"""

import argparse
from pathlib import Path
import sys
import yaml

from festivity.db import create_database


def register_command(subparsers):
    """Register the init command."""
    parser = subparsers.add_parser(
        'init',
        help='Initialize a new festivity workspace',
        description='Create workspace directory structure and configuration'
    )
    
    parser.add_argument(
        'workspace_path',
        type=Path,
        help='Path where workspace will be created'
    )
    
    parser.add_argument(
        '--dinov3-repo',
        type=Path,
        default='./dinov3',
        help='Path to local DINOv3 repository (default: ./dinov3)'
    )
    
    parser.add_argument(
        '--weights-source',
        type=Path,
        help='Path to directory containing DINOv3 weights to copy (optional)'
    )
    
    parser.set_defaults(func=execute)


def create_default_config(dinov3_repo: Path) -> dict:
    """Create default configuration dictionary."""
    return {
        'dinov3': {
            'model_type': 'dinov3_vits16',
            'weights_dir': 'weights',
            'repo_path': str(dinov3_repo)
        },
        'gps': {
            'offset_distance_left_m': 30.0,
            'offset_distance_right_m': 20.0,
            'min_heading_distance_m': 2.0,
            'address_match_distance_m': 30.0
        },
        'image_patterns': {
            'left_pattern': '_left',
            'right_pattern': '_right'
        },
        'training': {
            'optimal_c': 0.1,
            'patch_size': 16,
            'image_size': 768
        },
        'visualization': {
            'default_map_output': 'outputs/festivity_map.html'
        }
    }


def execute(args):
    """Execute the init command."""
    workspace_path = args.workspace_path.expanduser().absolute()
    
    # Check if workspace already exists
    if workspace_path.exists():
        response = input(f"Workspace {workspace_path} already exists. Continue? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Initialization cancelled.")
            sys.exit(0)
    
    # Create workspace directory
    print(f"Creating workspace at: {workspace_path}")
    workspace_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (workspace_path / 'images').mkdir(exist_ok=True)
    (workspace_path / 'models').mkdir(exist_ok=True)
    (workspace_path / 'weights').mkdir(exist_ok=True)
    (workspace_path / 'outputs').mkdir(exist_ok=True)
    print("  ✓ Created directory structure")
    
    # Resolve DINOv3 repo path to absolute (store absolute path in config)
    dinov3_repo_path = args.dinov3_repo.expanduser().absolute()

    # Create configuration file
    config_path = workspace_path / 'config.yaml'
    config = create_default_config(dinov3_repo_path)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  ✓ Created configuration file: {config_path}")
    
    # Create database
    db_path = workspace_path / 'database.db'
    create_database(db_path)
    print(f"  ✓ Created database: {db_path}")
    
    # Copy weights if specified
    if args.weights_source:
        weights_source = args.weights_source.expanduser().absolute()
        weights_dest = workspace_path / 'weights'
        
        if not weights_source.exists():
            print(f"  ⚠ Warning: Weights source directory not found: {weights_source}")
        elif not weights_source.is_dir():
            print(f"  ⚠ Warning: Weights source is not a directory: {weights_source}")
        else:
            import shutil
            weight_files = list(weights_source.glob('*.pth'))
            
            if not weight_files:
                print(f"  ⚠ Warning: No .pth weight files found in {weights_source}")
            else:
                for weight_file in weight_files:
                    dest_file = weights_dest / weight_file.name
                    shutil.copy2(weight_file, dest_file)
                    print(f"    • Copied {weight_file.name}")
                print(f"  ✓ Copied {len(weight_files)} weight file(s)")
    
    # Print next steps
    print("\n" + "="*60)
    print("Workspace initialized successfully!")
    print("="*60)
    print(f"\nWorkspace location: {workspace_path}")
    print(f"\nNext steps:")
    print(f"  1. Review and edit configuration: {config_path}")
    print(f"     - Adjust DINOv3 model settings")
    print(f"     - Configure GPS offset distance")
    print(f"     - Set image naming patterns")
    print(f"")
    print(f"  2. (Optional) Set environment variable:")
    print(f"     export FESTIVITY_WORKSPACE={workspace_path}")
    print(f"")
    print(f"  3. Train a model:")
    print(f"     festivity train --workspace {workspace_path} \\")
    print(f"       --images-dir /path/to/training/images \\")
    print(f"       --labels-dir /path/to/training/labels")
    print(f"")
    print(f"  4. Start processing images:")
    print(f"     festivity add-images --workspace {workspace_path} --images-dir /path/to/images")
    print(f"     festivity extract-gps --workspace {workspace_path}")
    print(f"     festivity score --workspace {workspace_path}")
    print(f"     festivity map --workspace {workspace_path}")
    print("")
