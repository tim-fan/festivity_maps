"""
Utility functions for festivity workspace management.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


def get_workspace_path(args: Any) -> Path:
    """
    Get workspace path from arguments or environment variable.
    
    Args:
        args: Argument namespace with optional 'workspace' attribute
        
    Returns:
        Path to workspace directory
        
    Raises:
        SystemExit: If workspace not specified or doesn't exist
    """
    workspace = None
    
    # Try to get from args first
    if hasattr(args, 'workspace') and args.workspace:
        workspace = args.workspace
    # Fall back to environment variable
    elif 'FESTIVITY_WORKSPACE' in os.environ:
        workspace = os.environ['FESTIVITY_WORKSPACE']
    else:
        print("Error: No workspace specified.", file=sys.stderr)
        print("Please provide --workspace argument or set FESTIVITY_WORKSPACE environment variable.", 
              file=sys.stderr)
        sys.exit(1)
    
    workspace_path = Path(workspace).expanduser().absolute()
    
    if not workspace_path.exists():
        print(f"Error: Workspace directory does not exist: {workspace_path}", file=sys.stderr)
        sys.exit(1)
    
    if not workspace_path.is_dir():
        print(f"Error: Workspace path is not a directory: {workspace_path}", file=sys.stderr)
        sys.exit(1)
    
    return workspace_path


def ensure_workspace_initialized(workspace_path: Path) -> None:
    """
    Validate that workspace has required structure.
    
    Args:
        workspace_path: Path to workspace directory
        
    Raises:
        SystemExit: If workspace is not properly initialized
    """
    required_items = {
        'config.yaml': 'file',
        'database.db': 'file',
        'images': 'dir',
        'models': 'dir',
        'outputs': 'dir',
    }
    
    missing = []
    for item, item_type in required_items.items():
        item_path = workspace_path / item
        if not item_path.exists():
            missing.append(item)
        elif item_type == 'file' and not item_path.is_file():
            missing.append(f"{item} (not a file)")
        elif item_type == 'dir' and not item_path.is_dir():
            missing.append(f"{item} (not a directory)")
    
    if missing:
        print(f"Error: Workspace not properly initialized: {workspace_path}", file=sys.stderr)
        print(f"Missing: {', '.join(missing)}", file=sys.stderr)
        print(f"\nPlease run: festivity init {workspace_path}", file=sys.stderr)
        sys.exit(1)


def load_config(workspace_path: Path) -> Dict[str, Any]:
    """
    Load configuration from workspace.
    
    Args:
        workspace_path: Path to workspace directory
        
    Returns:
        Configuration dictionary
        
    Raises:
        SystemExit: If config file doesn't exist or is invalid
    """
    config_path = workspace_path / 'config.yaml'
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            config = {}
        
        return config
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in config file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)


def detect_image_side(filename: str, config: Dict[str, Any]) -> Optional[bool]:
    """
    Detect whether image is from left or right camera based on filename.
    
    Args:
        filename: Image filename
        config: Configuration dictionary with image_patterns
        
    Returns:
        True if left camera, False if right camera, None if pattern not found
    """
    left_pattern = config.get('image_patterns', {}).get('left_pattern', '_left')
    right_pattern = config.get('image_patterns', {}).get('right_pattern', '_right')
    
    if left_pattern in filename:
        return True
    elif right_pattern in filename:
        return False
    else:
        return None


def resolve_path(path: str, workspace_root: Path, make_relative: bool = True) -> str:
    """
    Convert between absolute and workspace-relative paths.
    
    Args:
        path: Path to convert
        workspace_root: Workspace root directory
        make_relative: If True, convert to relative; if False, convert to absolute
        
    Returns:
        Converted path as string
    """
    path_obj = Path(path)
    
    if make_relative:
        # Convert absolute to relative
        if path_obj.is_absolute():
            try:
                return str(path_obj.relative_to(workspace_root))
            except ValueError:
                # Path is outside workspace, return absolute
                return str(path_obj.absolute())
        else:
            return str(path_obj)
    else:
        # Convert relative to absolute
        if path_obj.is_absolute():
            return str(path_obj)
        else:
            return str((workspace_root / path_obj).absolute())


def get_model_path(workspace_path: Path, config: Dict[str, Any]) -> Path:
    """
    Get the path to the trained classifier model.
    
    Args:
        workspace_path: Path to workspace directory
        config: Configuration dictionary
        
    Returns:
        Path to model file
    """
    return workspace_path / 'models' / 'festivity_classifier.pkl'


def get_database_path(workspace_path: Path) -> Path:
    """
    Get the path to the database file.
    
    Args:
        workspace_path: Path to workspace directory
        
    Returns:
        Path to database file
    """
    return workspace_path / 'database.db'
