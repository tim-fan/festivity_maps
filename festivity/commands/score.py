"""Score images with festivity model."""
import sys
import pickle
from pathlib import Path
import torch
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments

from festivity.utils import get_workspace_path, ensure_workspace_initialized, load_config, get_model_path, resolve_path
from festivity.db import get_db_connection, insert_festivity_scores, get_images_with_gps


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PATCH_SIZE = 16

MODEL_TO_NUM_LAYERS = {
    'dinov3_vits16': 12,
    'dinov3_vitb16': 12,
    'dinov3_vitl16': 24,
}


def register_command(subparsers):
    """Register the score command."""
    parser = subparsers.add_parser(
        'score',
        help='Score images with festivity model'
    )
    parser.add_argument(
        '--workspace',
        type=str,
        help='Path to workspace directory'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Rescore all images, even those already scored (default: skip existing)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit scoring to N random images (for testing)'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Score a single image by path (relative to workspace/images or absolute)'
    )
    parser.set_defaults(func=execute)


def resize_transform(img: Image.Image, target_size: int = 768) -> torch.Tensor:
    """Resize image maintaining aspect ratio, then convert to tensor."""
    w, h = img.size
    if w < h:
        new_w = target_size
        new_h = int(target_size * h / w)
    else:
        new_h = target_size
        new_w = int(target_size * w / h)
    
    img_resized = img.resize((new_w, new_h), Image.BICUBIC)
    return T.ToTensor()(img_resized)


def extract_features_for_image(model, img_tensor: torch.Tensor, model_name: str, device: str) -> tuple[torch.Tensor, tuple[int, int]]:
    """Extract DINOv3 features for a single image.
    
    Returns:
        features: [num_patches, feature_dim] tensor
        patch_shape: (h_patches, w_patches) number of patches in each dimension
    """
    n_layers = MODEL_TO_NUM_LAYERS[model_name]
    img_normalized = TF.normalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    with torch.inference_mode():
        with torch.autocast(device_type=device if device != 'cpu' else 'cpu', dtype=torch.float32):
            feats = model.get_intermediate_layers(
                img_normalized.unsqueeze(0).to(device),
                n=range(n_layers),
                reshape=True,
                norm=True
            )
            x = feats[-1].squeeze().detach().cpu()
            dim = x.shape[0]
            x = x.view(dim, -1).permute(1, 0)  # [num_patches, feature_dim]
    
    # Calculate patch dimensions from resized image
    h_patches = int(img_tensor.shape[1] / PATCH_SIZE)
    w_patches = int(img_tensor.shape[2] / PATCH_SIZE)
    
    return x, (h_patches, w_patches)


def create_heatmap(probabilities: np.ndarray, patch_shape: tuple[int, int]) -> np.ndarray:
    """Create a festivity heatmap as 8-bit grayscale array.
    
    Args:
        probabilities: [num_patches] array of festivity probabilities
        patch_shape: (h_patches, w_patches) number of patches
        
    Returns:
        8-bit grayscale heatmap array [h_patches, w_patches]
    """
    h_patches, w_patches = patch_shape
    
    # Reshape probabilities to patch grid
    # Note: reshape fills row-by-row, so we need to be careful about the order
    # The features come out in row-major order (h, w), matching the image layout
    fg_score = probabilities.reshape(h_patches, w_patches)
    
    # Convert to 8-bit (0-255 range)
    heatmap_8bit = (fg_score * 255).astype(np.uint8)
    
    return heatmap_8bit


def score_image(image_path, clf, model, model_name, device, image_size=768):
    """Score a single image and return mean festivity probability and patch predictions.
    
    Returns:
        mean_proba: Mean festivity probability across all patches
        probabilities: [num_patches] array of festivity probabilities per patch
        patch_shape: (h_patches, w_patches) number of patches
    """
    try:
        # Load image (already normalized in workspace, no EXIF transpose needed)
        img = Image.open(image_path).convert("RGB")
        img_resized = resize_transform(img, target_size=image_size)
        
        # Extract features
        features, patch_shape = extract_features_for_image(model, img_resized, model_name, device)
        
        # Get predictions from classifier
        proba = clf.predict_proba(features.numpy())[:, 1]  # Probability of festive class
        mean_proba = float(proba.mean())
        
        return mean_proba, proba, patch_shape
        
    except Exception as e:
        print(f"Error scoring {image_path.name}: {e}")
        return None, None, None


def execute(args):
    """Execute the score command."""
    workspace = get_workspace_path(args)
    ensure_workspace_initialized(workspace)
    
    config = load_config(workspace)
    
    # Check if classifier exists
    # model path is stored under workspace/models/festivity_classifier.pkl
    classifier_path = workspace / 'models' / 'festivity_classifier.pkl'
    if not classifier_path.exists():
        print(f"Error: Classifier not found at {classifier_path}")
        print("Please run 'festivity train' first to train a classifier")
        return 1
    
    # Load classifier
    print(f"Loading classifier from {classifier_path}")
    with open(classifier_path, 'rb') as f:
        classifier = pickle.load(f)
    
    # Get images to score
    conn = get_db_connection(workspace)
    
    if args.image:
        # Score a single image
        images_dir = workspace / 'images'
        img_path = Path(args.image)
        
        # Handle relative or absolute paths
        if not img_path.is_absolute():
            img_path = images_dir / img_path
        
        if not img_path.exists():
            print(f"Error: Image not found: {img_path}")
            conn.close()
            return 1
        
        # Use just the filename for database lookup
        all_filenames = {img_path.name}
        print(f"Scoring single image: {img_path.name}")
        
    elif args.force:
        # Score all images with GPS data
        cursor = conn.execute("SELECT filename FROM gps_data")
        all_filenames = {row[0] for row in cursor.fetchall()}
    else:
        # Score only images not yet in festivity_scores table
        images_with_gps_records = get_images_with_gps(conn)
        images_with_gps = {record['filename'] for record in images_with_gps_records}
        cursor = conn.execute("SELECT filename FROM festivity_scores")
        already_scored = {row[0] for row in cursor.fetchall()}
        all_filenames = images_with_gps - already_scored
    
    if not all_filenames:
        if args.force:
            print("No images with GPS data found in database")
        else:
            print("All images already scored. Use --force to rescore.")
        conn.close()
        return 0
    
    # Limit to N random images if requested (not compatible with --image)
    if args.limit and args.limit < len(all_filenames) and not args.image:
        import random
        all_filenames = set(random.sample(sorted(all_filenames), args.limit))
        print(f"  Limiting to {args.limit} random images for testing")
    
    print(f"\nScoring {len(all_filenames)} images...")
    
    # Load DINOv3 model
    model_type = config.get('dinov3', {}).get('model_type', 'dinov3_vits16')
    repo_path_config = config.get('dinov3', {}).get('repo_path', './dinov3')
    
    # Resolve repo path (allow absolute path stored in config)
    repo_path_str = resolve_path(repo_path_config, workspace, make_relative=False)
    repo_path = Path(repo_path_str)
    
    if not repo_path.exists():
        print(f"Error: DINOv3 repository not found at {repo_path}")
        print(f"  Configured path: {repo_path_config}")
        print(f"  Please update 'dinov3.repo_path' in config.yaml")
        return 1
    
    image_size = config.get('training', {}).get('image_size', 768)
    
    print(f"\nLoading DINOv3 model: {model_type}")
    print(f"  Repo path: {repo_path}")
    
    # Add dinov3 to path and import, then attempt to load model
    sys.path.insert(0, str(repo_path))
    try:
        model = torch.hub.load(str(repo_path), model_type, source='local', pretrained=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure DINOv3 repository is available and weights are in workspace/weights/")
        return 1

    # Determine device and try to use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")

    if device == 'cuda':
        try:
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        except Exception:
            pass

    # Move model to device; if OOM occurs, fall back to CPU
    try:
        model.to(device)
    except RuntimeError as e:
        print(f"Warning: Failed to move model to {device}: {e}")
        if device == 'cuda':
            print("Falling back to CPU to avoid GPU OOM")
            device = 'cpu'
            model.to(device)

    model.eval()
    
    # Create output directory for heatmaps
    heatmaps_dir = workspace / 'outputs' / 'heatmaps'
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    
    # Score images
    images_dir = workspace / 'images'
    results = []
    errors = 0
    
    for filename in tqdm(sorted(all_filenames), desc="Scoring images"):
        img_path = images_dir / filename
        
        if not img_path.exists():
            print(f"Warning: Image file not found: {filename}")
            errors += 1
            continue
        
        mean_proba, probabilities, patch_shape = score_image(img_path, classifier, model, model_type, device, image_size)
        
        if mean_proba is not None:
            results.append({
                'filename': filename,
                'mean_probability': mean_proba
            })
            
            # Create and save heatmap as 8-bit grayscale PNG
            try:
                heatmap_array = create_heatmap(probabilities, patch_shape)
                heatmap_img = Image.fromarray(heatmap_array, mode='L')
                
                # Save heatmap with same stem as original
                heatmap_path = heatmaps_dir / f"{img_path.stem}_heatmap.png"
                heatmap_img.save(heatmap_path)
            except Exception as e:
                print(f"Warning: Failed to create heatmap for {filename}: {e}")
        else:
            errors += 1
        
        # Clear GPU cache periodically to avoid memory issues
        if device == 'cuda' and len(results) % 100 == 0:
            torch.cuda.empty_cache()
    
    # Insert scores into database
    if results:
        print("\nSaving scores to database...")
        insert_festivity_scores(conn, results)
    
    conn.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Scoring Summary")
    print("=" * 60)
    print(f"  Total images scored: {len(results)}")
    print(f"  Errors: {errors}")
    print(f"  Heatmaps saved to: {heatmaps_dir}")
    
    if results:
        probabilities = [r['mean_probability'] for r in results]
        print(f"\nFestivity Score Statistics:")
        print(f"  Mean: {np.mean(probabilities):.3f}")
        print(f"  Median: {np.median(probabilities):.3f}")
        print(f"  Min: {np.min(probabilities):.3f}")
        print(f"  Max: {np.max(probabilities):.3f}")
    
    print(f"\nNext step:")
    print(f"  festivity map --workspace {workspace}")
    
    return 0
