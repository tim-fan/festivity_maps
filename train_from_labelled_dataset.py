"""
Training a Foreground Segmentation Tool with DINOv3

This script trains a linear foreground segmentation model using DINOv3 features.
Based on the DINOv3 foreground segmentation notebook.
"""

import io
import os
import pickle
import tarfile
import urllib.request
import argparse
from pathlib import Path

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm


# Constants
PATCH_SIZE = 16
IMAGE_SIZE = 768
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_DINOV3_VITS = "dinov3_vits16"
MODEL_DINOV3_VITSP = "dinov3_vits16plus"
MODEL_DINOV3_VITB = "dinov3_vitb16"
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_DINOV3_VITHP = "dinov3_vith16plus"
MODEL_DINOV3_VIT7B = "dinov3_vit7b16"

MODEL_TO_NUM_LAYERS = {
    MODEL_DINOV3_VITS: 12,
    MODEL_DINOV3_VITSP: 12,
    MODEL_DINOV3_VITB: 12,
    MODEL_DINOV3_VITL: 24,
    MODEL_DINOV3_VITHP: 32,
    MODEL_DINOV3_VIT7B: 40,
}

IMAGES_URI = "https://dl.fbaipublicfiles.com/dinov3/notebooks/foreground_segmentation/foreground_segmentation_images.tar.gz"
LABELS_URI = "https://dl.fbaipublicfiles.com/dinov3/notebooks/foreground_segmentation/foreground_segmentation_labels.tar.gz"


def load_dataset_from_directories(images_dir: str, labels_dir: str) -> tuple[list[Image.Image], list[Image.Image]]:
    """Load images and labels from local directories.
    
    Only loads images that have corresponding labels.
    Matches by filename (e.g., IMG_123.jpg -> IMG_123.png).
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    # Find all label files
    label_files = sorted(labels_path.glob('*.png'))
    
    if not label_files:
        raise ValueError(f"No label files found in {labels_dir}")
    
    images = []
    labels = []
    
    for label_file in label_files:
        # Find corresponding image file (try common extensions)
        img_stem = label_file.stem
        img_file = None
        
        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
            candidate = images_path / f"{img_stem}{ext}"
            if candidate.exists():
                img_file = candidate
                break
        
        if img_file is None:
            print(f"Warning: No image found for label {label_file.name}, skipping")
            continue
        
        # Load image and label
        try:
            img = Image.open(img_file).convert('RGB')
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
        raise ValueError("No valid image/label pairs found")
    
    print(f"Loaded {len(images)} image/label pairs from {images_dir} and {labels_dir}")
    return images, labels


def load_images_from_remote_tar(tar_uri: str) -> list[Image.Image]:
    """Load images from a remote tar.gz file."""
    images = []
    with urllib.request.urlopen(tar_uri) as f:
        tar = tarfile.open(fileobj=io.BytesIO(f.read()))
        for member in tar.getmembers():
            image_data = tar.extractfile(member)
            image = Image.open(image_data)
            images.append(image)
    return images


def resize_transform(
    mask_image: Image.Image,
    image_size: int = IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    """Resize image to dimensions divisible by patch size."""
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))


def visualize_sample(images, labels, data_index=0, save_path=None):
    """Visualize a sample image/mask pair."""
    print(f"Showing image / mask at index {data_index}:")
    
    image = images[data_index]
    mask = labels[data_index]
    foreground = Image.composite(image, mask, mask)
    mask_bg_np = np.copy(np.array(mask))
    mask_bg_np[:, :, 3] = 255 - mask_bg_np[:, :, 3]
    mask_bg = Image.fromarray(mask_bg_np)
    background = Image.composite(image, mask_bg, mask_bg)

    data_to_show = [image, mask, foreground, background]
    data_labels = ["Image", "Mask", "Foreground", "Background"]

    plt.figure(figsize=(16, 4), dpi=300)
    for i in range(len(data_to_show)):
        plt.subplot(1, len(data_to_show), i + 1)
        plt.imshow(data_to_show[i])
        plt.axis('off')
        plt.title(data_labels[i], fontsize=12)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close()


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
        print(f'Validation using image_{i+1:02d}.jpg')
        train_selection = image_index != float(i)
        fold_x = xs[train_selection].numpy()
        fold_y = (ys[train_selection] > 0).long().numpy()
        val_x = xs[~train_selection].numpy()
        val_y = (ys[~train_selection] > 0).long().numpy()

        plt.figure()
        for j, c in enumerate(cs):
            print(f"Training logistic regression with C={c:.2e}")
            clf = LogisticRegression(random_state=0, C=c, max_iter=10000).fit(fold_x, fold_y)
            output = clf.predict_proba(val_x)
            precision, recall, thresholds = precision_recall_curve(val_y, output[:, 1])
            s = average_precision_score(val_y, output[:, 1])
            scores[i, j] = s
            plt.plot(recall, precision, label=f'C={c:.1e} AP={s*100:.1f}')

        plt.grid()
        plt.xlabel('recall')
        plt.title(f'image_{i+1:02d}.jpg')
        plt.ylabel('precision')
        plt.axis([0, 1, 0, 1])
        plt.legend()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'pr_curve_image_{i+1:02d}.png')
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved PR curve to {save_path}")
        else:
            plt.show()
        plt.close()
    
    return scores, cs


def plot_average_ap(scores, cs, save_path=None):
    """Plot average AP across all validation images for different C values."""
    plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(scores.mean(axis=0), marker='o')
    plt.xticks(np.arange(len(cs)), [f"{c:.0e}" for c in cs])
    plt.xlabel('Regularization C')
    plt.ylabel('Average AP')
    plt.title('Model Selection: Average AP vs Regularization')
    plt.grid()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved average AP plot to {save_path}")
    else:
        plt.show()
    plt.close()


def train_final_classifier(xs, ys, c=0.1):
    """Train the final classifier with the optimal C value."""
    print(f"\nTraining final classifier with C={c}")
    clf = LogisticRegression(random_state=0, C=c, max_iter=100000, verbose=1).fit(
        xs.numpy(), (ys > 0).long().numpy()
    )
    return clf


def test_classifier(clf, model, model_name, test_image_path, save_path=None):
    """Test the classifier on a test image."""
    print(f"\nTesting on image: {test_image_path}")
    
    # Load image
    if test_image_path.startswith('http'):
        with urllib.request.urlopen(test_image_path) as f:
            test_image = Image.open(f).convert("RGB")
    else:
        test_image = Image.open(test_image_path).convert("RGB")
    
    test_image_resized = resize_transform(test_image)
    test_image_normalized = TF.normalize(test_image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    n_layers = MODEL_TO_NUM_LAYERS[model_name]
    
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            feats = model.get_intermediate_layers(
                test_image_normalized.unsqueeze(0).cuda(), 
                n=range(n_layers), 
                reshape=True, 
                norm=True
            )
            x = feats[-1].squeeze().detach().cpu()
            dim = x.shape[0]
            x = x.view(dim, -1).permute(1, 0)

    h_patches, w_patches = [int(d / PATCH_SIZE) for d in test_image_resized.shape[1:]]

    fg_score = clf.predict_proba(x.numpy())[:, 1].reshape(h_patches, w_patches)
    fg_score_mf = torch.from_numpy(signal.medfilt2d(fg_score, kernel_size=3))

    plt.figure(figsize=(12, 4), dpi=150)
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(test_image_resized.permute(1, 2, 0))
    plt.title('Input Image')
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(fg_score)
    plt.title('Foreground Score')
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(fg_score_mf)
    plt.title('+ Median Filter')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved test result to {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train foreground segmentation with DINOv3')
    parser.add_argument('--model', type=str, default=MODEL_DINOV3_VITL,
                        choices=[MODEL_DINOV3_VITS, MODEL_DINOV3_VITSP, MODEL_DINOV3_VITB, 
                                MODEL_DINOV3_VITL, MODEL_DINOV3_VITHP, MODEL_DINOV3_VIT7B],
                        help='DINOv3 model to use')
    parser.add_argument('--dinov3-location', type=str, default='./dinov3',
                        help='Path to local DINOv3 repository')
    parser.add_argument('--images-dir', type=str, default=None,
                        help='Directory containing training images (if using local dataset)')
    parser.add_argument('--labels-dir', type=str, default=None,
                        help='Directory containing label masks (if using local dataset)')
    parser.add_argument('--use-remote-dataset', action='store_true',
                        help='Use the remote example dataset instead of local dataset')
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Directory to save outputs')
    parser.add_argument('--model-save-path', type=str, default='./fg_classifier.pkl',
                        help='Path to save the trained classifier')
    parser.add_argument('--test-image', type=str, default=None,
                        help='Path or URL to test image (if not provided, uses first image from dataset)')
    parser.add_argument('--optimal-c', type=float, default=0.1,
                        help='Optimal C value for final classifier training')
    parser.add_argument('--skip-cv', action='store_true',
                        help='Skip cross-validation and train directly with optimal C')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading DINOv3 model: {args.model}")
    print(f"DINOv3 location: {args.dinov3_location}")
    
    model = torch.hub.load(
        repo_or_dir=args.dinov3_location,
        model=args.model,
        source="local",
    )
    model.cuda()
    print("Model loaded successfully")
    
    # Load training data
    print("\nLoading training data...")
    if args.use_remote_dataset:
        # Use remote example dataset
        images = load_images_from_remote_tar(IMAGES_URI)
        labels = load_images_from_remote_tar(LABELS_URI)
        n_images = len(images)
        assert n_images == len(labels), f"{len(images)=}, {len(labels)=}"
        print(f"Loaded {n_images} images and labels from remote dataset")
        # Default test image for remote dataset
        if args.test_image is None:
            args.test_image = 'https://dl.fbaipublicfiles.com/dinov3/notebooks/foreground_segmentation/test_image.jpg'
    else:
        # Use local dataset
        if args.images_dir is None or args.labels_dir is None:
            # Default to the user's dataset
            args.images_dir = args.images_dir or os.path.expanduser('~/projects/2024/1203_festivity_map/datasets/20241204_low_exposure/images')
            args.labels_dir = args.labels_dir or os.path.expanduser('~/projects/2024/1203_festivity_map/datasets/20241204_low_exposure/labels')
        
        images, labels = load_dataset_from_directories(args.images_dir, args.labels_dir)
        n_images = len(images)
        
        # Save first image temporarily for testing if no test image specified
        if args.test_image is None:
            test_image_path = os.path.join(args.output_dir, 'test_image_from_dataset.jpg')
            images[0].save(test_image_path)
            args.test_image = test_image_path
            print(f"Using first training image as test image: {args.test_image}")
    
    # Visualize sample
    visualize_sample(images, labels, data_index=0, 
                     save_path=os.path.join(args.output_dir, 'sample_visualization.png'))
    
    # Extract features and labels
    print("\nExtracting features and labels...")
    xs, ys, image_index = extract_features_and_labels(images, labels, model, args.model)
    
    # Cross-validation or direct training
    if not args.skip_cv:
        print("\nPerforming cross-validation...")
        scores, cs = train_with_cross_validation(xs, ys, image_index, n_images, 
                                                  save_dir=os.path.join(args.output_dir, 'pr_curves'))
        
        # Plot average AP
        plot_average_ap(scores, cs, save_path=os.path.join(args.output_dir, 'average_ap.png'))
        
        # Find best C
        best_c_idx = scores.mean(axis=0).argmax()
        best_c = cs[best_c_idx]
        print(f"\nBest C from cross-validation: {best_c:.2e}")
        print(f"Using optimal C from args: {args.optimal_c}")
    
    # Train final classifier
    clf = train_final_classifier(xs, ys, c=args.optimal_c)
    
    # Save classifier
    with open(args.model_save_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nClassifier saved to {args.model_save_path}")
    
    # Test on sample image
    if args.test_image:
        test_classifier(clf, model, args.model, args.test_image,
                       save_path=os.path.join(args.output_dir, 'test_result.png'))
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
