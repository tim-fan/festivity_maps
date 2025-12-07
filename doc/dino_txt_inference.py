

# based on
#  https://visionbrick.com/introduction-to-dinov3-generating-similarity-maps-with-vision-transformers/
#  https://github.com/facebookresearch/dinov3/blob/main/notebooks/dinotxt_segmentation_inference.ipynb
# 
# Never got it working properly, saved here under doc/ for future debugging

import dataclasses
import math
import warnings
from typing import Callable
import os
import random

import glob
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

import lovely_tensors
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVTF
import tqdm
from omegaconf import OmegaConf
from torch import Tensor, nn
from torchmetrics.classification import MulticlassJaccardIndex
from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l

def encode_image(model, img: Tensor) -> tuple[Tensor, Tensor]:
    """Extract image features from the backbone and the additional blocks."""
    B, _, H, W = img.shape
    P = model.visual_model.backbone.patch_size # In the case of our DINOv3
    new_H = math.ceil(H / P) * P
    new_W = math.ceil(W / P) * P

    # Stretch image to a multiple of patch size
    if (H, W) != (new_H, new_W):
        img = F.interpolate(img, size=(new_H, new_W), mode="bicubic", align_corners=False)  # [B, 3, H', W']

    B, _, h_i, w_i = img.shape

    backbone_patches = None
    cls_tokens, _, patch_tokens = model.visual_model.get_class_and_patch_tokens(img)
    blocks_patches = (
        patch_tokens.reshape(B, h_i // P, w_i // P, -1).contiguous()
    ) # [1, h, w, D]

    return backbone_patches, blocks_patches

class ShortSideResize(nn.Module):
    def __init__(self, size: int, interpolation: TVT.InterpolationMode) -> None:
        super().__init__()
        self.size = size
        self.interpolation = interpolation

    def forward(self, img: Tensor) -> Tensor:
        _, h, w = TVTF.get_dimensions(img)
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img
        if w < h:
            new_w = self.size
            new_h = int(self.size * h / w)
            return TVTF.resize(img, [new_h, new_w], self.interpolation)
        else:
            new_h = self.size
            new_w = int(self.size * w / h)
            return TVTF.resize(img, [new_h, new_w], self.interpolation)
        

def predict_whole(model, img: Tensor, text_features: Tensor) -> Tensor:
    # Extract image features from the additional blocks, ignore the backbone features
    _, H, W = img.shape
    _, blocks_feats = encode_image(model, img.unsqueeze(0))  # [1, h, w, D]
    _, h, w, _ = blocks_feats.shape
    blocks_feats = blocks_feats.squeeze(0)  # [h, w, D]

    # Cosine similarity between patch features and text features (already normalized)
    blocks_feats = F.normalize(blocks_feats, p=2, dim=-1)  # [h, w, D]


    cos = torch.einsum("cd,hwd->chw", text_features, blocks_feats)  # [num_classes, h, w]

    # Return low-res cosine similarities, they will be upsampled to the target resolution later
    return cos.squeeze(0) # [H, W]



class ZeroShotSegmentationDataset(torch.utils.data.Dataset):
    CLASS_NAMES: tuple[str, ...]
    IGNORE_ZERO_LABEL: bool  # If True, map label 0 to 255 so it's ignored, and shift all other labels by -1
    transform: Callable[[PIL.Image.Image], Tensor]

    def __init__(self, transform: Callable[[PIL.Image.Image], Tensor]) -> None:
        self.transform = transform

    def _mask_to_tensor(self, mask_pil: PIL.Image.Image) -> Tensor:
        mask = torch.from_numpy(np.array(mask_pil)).long()
        if self.IGNORE_ZERO_LABEL:
            mask = torch.where((mask == 0) | (mask == 255), 255, mask - 1)
        return mask

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img, target = self.ds[idx]
        img = self.transform(img)
        target = self._mask_to_tensor(target)
        return img, target

    def __len__(self) -> int:
        return len(self.ds)

# gemini generated, folder of unlabeled images
class UnlabeledImageFolder(Dataset):
    def __init__(self, root_dir, transform=None, valid_extensions=("*.jpg", "*.jpeg", "*.png")):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        
        # Recursive search for images
        for ext in valid_extensions:
            self.image_paths.extend(list(self.root_dir.rglob(ext)))
            
        self.image_paths = sorted(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Convert to RGB to handle occasional alpha channels or grayscale
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        # Return image and the stem (filename without extension) for saving
        return img, img_path.stem

# --- Configuration ---

@dataclasses.dataclass
class Configuration:
    resize: int = 512  # Short side of the input images

# Local setup
lovely_tensors.monkey_patch()
warnings.filterwarnings("ignore", message="xFormers")
cfg: Configuration = OmegaConf.to_object(
    OmegaConf.structured(Configuration),
)
print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

image_dir = "../datasets/20241204_low_exposure/images/"
output_dir = "../datasets/20241204_low_exposure/inference_results"
os.makedirs(output_dir, exist_ok=True)

# Define classes manually since we have no dataset metadata
class_names = ["festive_lights"] 

# --- Dataset & Loader ---
NORMALIZE_IMAGENET = TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = TVT.Compose(
    [
        ShortSideResize(cfg.resize, TVT.InterpolationMode.BICUBIC),
        TVT.ToTensor(),
        NORMALIZE_IMAGENET,
    ]
)

dataset = UnlabeledImageFolder(image_dir, transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1, 
    num_workers=4,
    shuffle=False,
    pin_memory=True
)

print(f"Found {len(dataset)} images. Classes: {class_names}")


# model takes roughly 3.2 GB gpu memory
model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(
    weights="weights/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
    backbone_weights="weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")

# Force CPU for testing
device = "cpu"

model.to(device, non_blocking=True)
if device == "cpu":
    model.float()
model.eval()
tokenizer = tokenizer.tokenize

text_feats = None # TODO
text_feats = []
# text = ["a nightime photo of a festive light making up part of a house's holiday decoration"]
text = ["festive house decoration"]
tokens = tokenizer(text).to(device, non_blocking=True)
feats = model.encode_text(tokens)  # [num_prompts, 2D]
feats = feats[:, feats.shape[1] // 2 :]  # The 1st half of the features corresponds to the CLS token, drop it
feats = F.normalize(feats, p=2, dim=-1)  # Normalize each text embedding
feats = feats.mean(dim=0)  # Average over all prompt embeddings per class
feats = F.normalize(feats, p=2, dim=-1)  # Normalize again
text_feats.append(feats)
text_feats = torch.stack(text_feats)  # [num_classes, D]
print(f"Text features: {text_feats}")

# Randomly select 20 images from the dataset
num_samples = min(20, len(dataset))
random_indices = random.sample(range(len(dataset)), num_samples)
print(f"Processing {num_samples} randomly selected images out of {len(dataset)} total images")

model.eval()
with torch.no_grad():
    for idx in tqdm.tqdm(random_indices, desc="Inference"):
        img, stem = dataset[idx]
        img_input = img 
        
        # pred shape is now [h, w] (raw cosine scores, e.g., -0.3 to 0.6)
        pred = predict_whole(model, img_input, text_feats)

        # 1. Normalize scores to the 0-1 range
        # Use a large range (e.g., -0.2 to 0.5) to capture differences, 
        # or use min/max of the current tensor for full range utilization.
        min_val = pred.min()
        max_val = pred.max()

        # Clamp min/max for more consistent visualization across images (optional)
        # min_val = torch.tensor(-0.1) 
        # max_val = torch.tensor(0.4) 
        
        # Normalized score: [0.0, 1.0]
        norm_pred = (pred - min_val) / (max_val - min_val) 

        # 2. Scale to 0-255 and convert to 8-bit integer
        heatmap_array = (norm_pred * 255).cpu().numpy().astype(np.uint8)

        # 3. Interpolate (Resize) back to the original resolution if needed
        # (You omitted this step in your previous loop, but it's essential 
        # for a full-resolution heatmap image.)
        # If original resolution H_orig, W_orig is available:
        # heatmap_tensor = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=(H_orig, W_orig), mode="bilinear").squeeze()
        
        # --- Save Result ---
        # Get original image dimensions
        original_img = dataset.image_paths[idx]
        original_pil = Image.open(original_img).convert("RGB")
        H_orig, W_orig = original_pil.size[1], original_pil.size[0]  # PIL uses (W, H)

        # Upscale heatmap to original resolution
        heatmap_tensor = F.interpolate(
            norm_pred.unsqueeze(0).unsqueeze(0), 
            size=(H_orig, W_orig), 
            mode="bilinear",
            align_corners=False
        ).squeeze()

        # Convert upscaled heatmap to 8-bit array
        heatmap_array_upscaled = (heatmap_tensor * 255).cpu().numpy().astype(np.uint8)
        heatmap_img_upscaled = Image.fromarray(heatmap_array_upscaled, mode='L')

        # Convert grayscale heatmap to RGB for concatenation
        heatmap_rgb = heatmap_img_upscaled.convert('RGB')

        # Concatenate original image and heatmap side by side
        combined_img = Image.new('RGB', (W_orig * 2, H_orig))
        combined_img.paste(original_pil, (0, 0))
        combined_img.paste(heatmap_rgb, (W_orig, 0))

        # Save combined image
        save_path = os.path.join(output_dir, f"{stem}_similarity_combined.png")
        combined_img.save(save_path)

        # Also save just the upscaled heatmap
        # heatmap_path = os.path.join(output_dir, f"{stem}_similarity.png")
        # heatmap_img_upscaled.save(heatmap_path)


