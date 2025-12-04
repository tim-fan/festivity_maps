"""Add images command - copy/move images to workspace."""
import os
import shutil
from pathlib import Path
from festivity.utils import get_workspace_path, ensure_workspace_initialized, detect_image_side
from festivity.db import get_db_connection
from PIL import Image, ImageOps
from tqdm import tqdm


def register_command(subparsers):
    """Register the add-images command."""
    parser = subparsers.add_parser(
        'add-images',
        help='Add images to the workspace'
    )
    parser.add_argument(
        '--workspace',
        type=str,
        help='Path to workspace directory'
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        required=True,
        help='Directory containing images to add'
    )
    parser.add_argument(
        '--move',
        action='store_true',
        help='Move images instead of copying (default: copy)'
    )
    parser.add_argument(
        '--skip-duplicates',
        action='store_true',
        help='Skip images that already exist in workspace'
    )
    parser.set_defaults(func=execute)


def execute(args):
    """Execute the add-images command."""
    workspace = get_workspace_path(args)
    ensure_workspace_initialized(workspace)
    
    # Load config
    from festivity.utils import load_config
    config = load_config(workspace)
    
    images_dir = Path(args.images_dir).expanduser().resolve()
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return 1
    
    workspace_images = workspace / 'images'
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    # Find all images
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f'*{ext}'))
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return 1
    
    print(f"\nAdding images to workspace: {workspace}")
    print(f"Source: {images_dir}")
    print(f"Mode: {'move' if args.move else 'copy'}")
    print(f"Found {len(image_files)} images")
    
    # Copy or move images with orientation normalization
    copied_files = copy_or_move_images(images_dir, workspace_images, not args.move, config)
    
    if copied_files is None:
        return 1
    
    print(f"\nNext step:")
    print(f"  festivity extract-gps --workspace {workspace}")
    
    return 0


def copy_or_move_images(src_dir: Path, dest_dir: Path, copy: bool, config: dict):
    """Copy or move images from source to workspace, normalizing orientation."""
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    # Find all images
    image_files = []
    for ext in image_extensions:
        image_files.extend(src_dir.glob(f'*{ext}'))
    
    if not image_files:
        print(f"No images found in {src_dir}")
        return
    
    # Initialize counters
    copied_files = []
    side_counts = {True: 0, False: 0, None: 0}
    errors = 0
    
    print(f"\n{'Copying' if copy else 'Moving'} images:")
    
    for src_file in tqdm(image_files, desc=f"{'Copying' if copy else 'Moving'} images"):
        dest_file = dest_dir / src_file.name
        
        try:
            # Open image and normalize orientation
            img = Image.open(src_file)
            
            # Get EXIF data before transformation
            exif = img.getexif()
            
            # Apply EXIF rotation
            img = ImageOps.exif_transpose(img)
            
            # Remove orientation tag from EXIF (since we've applied it)
            if exif and 274 in exif:  # 274 is the Orientation tag
                exif[274] = 1  # Set to normal orientation
            
            # Save normalized image with preserved EXIF (minus rotation)
            if dest_file.suffix.lower() in ['.jpg', '.jpeg']:
                img.save(dest_file, quality=95, exif=exif if exif else None)
            else:
                img.save(dest_file)
            
            # Detect side
            is_left = detect_image_side(src_file.name, config)
            if is_left is not None:
                side_counts[is_left] += 1
            
            copied_files.append({
                'filename': src_file.name,
                'is_left': is_left
            })
            
            # Remove source if moving
            if not copy:
                src_file.unlink()
                
        except Exception as e:
            print(f"Error processing {src_file.name}: {e}")
            errors += 1
    
    print(f"\nProcessed {len(image_files)} images:")
    print(f"  Successfully {'copied' if copy else 'moved'}: {len(copied_files)}")
    print(f"  Errors: {errors}")
    print(f"  Left side: {side_counts[True]}")
    print(f"  Right side: {side_counts[False]}")
    print(f"  Side unknown: {side_counts[None]}")
    
    return copied_files
