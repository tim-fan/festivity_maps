"""
Train festivity classifier.

Reference https://github.com/facebookresearch/dinov3/blob/main/notebooks/foreground_segmentation.ipynb
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

from festivity.utils import get_workspace_path, ensure_workspace_initialized, load_config, get_model_path


# Constants
PATCH_SIZE = 16
IMAGE_SIZE = 768
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_TO_NUM_LAYERS = {
    "dinov3_vits16": 12,
    "dinov3_vits16plus": 12,
    "dinov3_vitb16": 12,
    "dinov3_vitl16": 24,
    "dinov3_vith16plus": 32,
    "dinov3_vit7b16": 40,
}


def register_command(subparsers):
    """Register the train command."""
    parser = subparsers.add_parser(
        'train',
        help='Train festivity classifier from labelled dataset',
        description='Train a classifier using DINOv3 features'
    )
    
    parser.add_argument(
        '--workspace',
        type=str,
        help='Workspace root path (or use FESTIVITY_WORKSPACE env var)'
    )
    
    parser.add_argument(
        '--images-dir',
        type=Path,
        required=True,
        help='Directory containing training images'
    )
    
    parser.add_argument(
        '--labels-dir',
        type=Path,
        required=True,
        help='Directory containing label masks (RGBA .png files)'
    )
    
    parser.add_argument(
        '--skip-cv',
        action='store_true',
        help='Skip cross-validation (use optimal_c from config directly)'
    )
    
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save PR curves and diagnostics to outputs/training/'
    )
    
    parser.set_defaults(func=execute)


def load_dataset_from_directories(images_dir: Path, labels_dir: Path):
    """Load images and labels from local directories."""
    # Find all label files
    label_files = sorted(labels_dir.glob('*.png'))
    
    if not label_files:
        print(f"Error: No label files found in {labels_dir}", file=sys.stderr)
        sys.exit(1)
    
    images = []
    labels = []
    
    for label_file in label_files:
        # Find corresponding image file
        img_stem = label_file.stem
        img_file = None
        
        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
            candidate = images_dir / f"{img_stem}{ext}"
            if candidate.exists():
                img_file = candidate
                break
        
        if img_file is None:
            print(f"Warning: No image found for label {label_file.name}, skipping")
            continue
        
        # Load image and label
        try:
            from PIL import ImageOps
            
            img = Image.open(img_file)
            img = img.convert('RGB')
            
            label = Image.open(label_file)
            
            # Validate label format
            if label.mode != 'RGBA':
                print(f"Warning: Label {label_file.name} is not RGBA mode, skipping")
                continue
            
            if img.size != label.size:
                print(f"Warning: Image and label size mismatch for {img_file.name}, skipping")
                continue
            
            images.append(img)
            labels.append(label)
            
        except Exception as e:
            print(f"Error loading {img_file.name} or {label_file.name}: {e}, skipping")
            continue
    
    if not images:
        print("Error: No valid image/label pairs found", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(images)} image/label pairs")
    return images, labels


def resize_transform(mask_image: Image.Image, image_size: int = IMAGE_SIZE, patch_size: int = PATCH_SIZE):
    """Resize image to dimensions divisible by patch size."""
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))


def extract_features_and_labels(images, labels, model, model_name, patch_size=PATCH_SIZE):
    """Extract DINOv3 features and quantized labels from images."""
    xs = []
    ys = []
    image_index = []
    
    n_images = len(images)
    n_layers = MODEL_TO_NUM_LAYERS[model_name]
    
    # Quantization filter for the given patch size
    patch_quant_filter = torch.nn.Conv2d(1, 1, patch_size, stride=patch_size, bias=False)
    patch_quant_filter.weight.data.fill_(1.0 / (patch_size * patch_size))
    
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i in tqdm(range(n_images), desc="Extracting features"):
                # Loading the ground truth
                mask_i = labels[i].split()[-1]
                mask_i_resized = resize_transform(mask_i)
                mask_i_quantized = patch_quant_filter(mask_i_resized).squeeze().view(-1).detach().cpu()
                ys.append(mask_i_quantized)
                
                # Loading the image data
                image_i = images[i].convert('RGB')
                image_i_resized = resize_transform(image_i)
                image_i_resized = TF.normalize(image_i_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
                image_i_resized = image_i_resized.unsqueeze(0).cuda()

                feats = model.get_intermediate_layers(image_i_resized, n=range(n_layers), reshape=True, norm=True)
                dim = feats[-1].shape[1]
                xs.append(feats[-1].squeeze().view(dim, -1).permute(1, 0).detach().cpu())

                image_index.append(i * torch.ones(ys[-1].shape))

    # Concatenate all lists into torch tensors
    xs = torch.cat(xs)
    ys = torch.cat(ys)
    image_index = torch.cat(image_index)

    # Keeping only the patches that have clear positive or negative label
    idx = (ys < 0.01) | (ys > 0.99)
    xs = xs[idx]
    ys = ys[idx]
    image_index = image_index[idx]

    print(f"Design matrix size: {xs.shape}")
    print(f"Label matrix size: {ys.shape}")
    
    return xs, ys, image_index


def train_with_cross_validation(xs, ys, image_index, n_images, save_dir=None):
    """Train classifiers with different C values using leave-one-out cross-validation."""
    cs = np.logspace(-7, 0, 8)
    scores = np.zeros((n_images, len(cs)))

    for i in range(n_images):
        # Leave-one-out: train on all but image i, validate on image i
        print(f'Cross-validation using image {i+1}/{n_images}')
        train_selection = image_index != float(i)
        fold_x = xs[train_selection].numpy()
        fold_y = (ys[train_selection] > 0).long().numpy()
        val_x = xs[~train_selection].numpy()
        val_y = (ys[~train_selection] > 0).long().numpy()

        plt.figure()
        for j, c in enumerate(cs):
            print(f"  Training with C={c:.2e}")
            clf = LogisticRegression(random_state=0, C=c, max_iter=10000).fit(fold_x, fold_y)
            output = clf.predict_proba(val_x)
            precision, recall, thresholds = precision_recall_curve(val_y, output[:, 1])
            s = average_precision_score(val_y, output[:, 1])
            scores[i, j] = s
            plt.plot(recall, precision, label=f'C={c:.1e} AP={s*100:.1f}')

        plt.grid()
        plt.xlabel('recall')
        plt.title(f'Image {i+1}/{n_images}')
        plt.ylabel('precision')
        plt.axis([0, 1, 0, 1])
        plt.legend()
        
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f'pr_curve_image_{i+1:02d}.png'
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.close()
    
    return scores, cs


def train_final_classifier(xs, ys, c=0.1):
    """Train the final classifier with the optimal C value."""
    print(f"\nTraining final classifier with C={c}")
    clf = LogisticRegression(random_state=0, C=c, max_iter=100000, verbose=1).fit(
        xs.numpy(), (ys > 0).long().numpy()
    )
    return clf


def execute(args):
    """Execute the train command."""
    # Get and validate workspace
    workspace_path = get_workspace_path(args)
    ensure_workspace_initialized(workspace_path)
    
    # Load config
    config = load_config(workspace_path)
    
    # Validate directories
    images_dir = args.images_dir.expanduser().absolute()
    labels_dir = args.labels_dir.expanduser().absolute()
    
    if not images_dir.exists() or not images_dir.is_dir():
        print(f"Error: Images directory not found: {images_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not labels_dir.exists() or not labels_dir.is_dir():
        print(f"Error: Labels directory not found: {labels_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Get model settings from config
    model_type = config.get('dinov3', {}).get('model_type', 'dinov3_vits16')
    repo_path = config.get('dinov3', {}).get('repo_path', './dinov3')
    optimal_c = config.get('training', {}).get('optimal_c', 0.1)
    
    print(f"\nTraining Configuration:")
    print(f"  Workspace: {workspace_path}")
    print(f"  DINOv3 model: {model_type}")
    print(f"  DINOv3 repo: {repo_path}")
    print(f"  Optimal C: {optimal_c}")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")
    print()
    
    # Load DINOv3 model
    print(f"Loading DINOv3 model: {model_type}")
    try:
        model = torch.hub.load(
            repo_or_dir=repo_path,
            model=model_type,
            source="local",
        )
        model.cuda()
        print("  ✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading DINOv3 model: {e}", file=sys.stderr)
        print(f"Make sure the DINOv3 repository is available at: {repo_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load training data
    print("\nLoading training data...")
    images, labels = load_dataset_from_directories(images_dir, labels_dir)
    n_images = len(images)
    
    # Extract features and labels
    print("\nExtracting features and labels...")
    xs, ys, image_index = extract_features_and_labels(images, labels, model, model_type)
    
    # Cross-validation or direct training
    final_c = optimal_c  # Default to config value
    
    if not args.skip_cv:
        print("\nPerforming cross-validation...")
        save_dir = workspace_path / 'outputs' / 'training' / 'pr_curves' if args.save_plots else None
        scores, cs = train_with_cross_validation(xs, ys, image_index, n_images, save_dir)
        
        # Find best C
        best_c_idx = scores.mean(axis=0).argmax()
        best_c = cs[best_c_idx]
        print(f"\nBest C from cross-validation: {best_c:.2e}")
        
        # Use the best C from cross-validation
        final_c = best_c
        print(f"Using C={final_c:.2e} for final classifier")
        
        if args.save_plots:
            # Plot average AP
            plt.figure(figsize=(6, 4), dpi=150)
            plt.plot(scores.mean(axis=0), marker='o')
            plt.xticks(np.arange(len(cs)), [f"{c:.0e}" for c in cs])
            plt.xlabel('Regularization C')
            plt.ylabel('Average AP')
            plt.title('Model Selection: Average AP vs Regularization')
            plt.grid()
            save_path = workspace_path / 'outputs' / 'training' / 'average_ap.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved average AP plot to {save_path}")
    else:
        print(f"\nSkipping cross-validation, using C={final_c:.2e} from config")
    
    # Train final classifier
    clf = train_final_classifier(xs, ys, c=final_c)
    
    # Save classifier
    model_path = get_model_path(workspace_path, config)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}")
    print(f"Classifier saved to: {model_path}")
    print(f"Trained on {n_images} images")
    print(f"\nNext steps:")
    print(f"  1. Add images: festivity add-images --workspace {workspace_path} --images-dir /path/to/images")
    print(f"  2. Extract GPS: festivity extract-gps --workspace {workspace_path}")
    print(f"  3. Score images: festivity score --workspace {workspace_path}")
    print(f"  4. Generate map: festivity map --workspace {workspace_path}")
    print()
