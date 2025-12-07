# Creating Label Masks in GIMP for DINOv3 Training

This guide explains how to create label mask files in the correct format for training the DINOv3 foreground segmentation model.

## Required Label Format

- **Mode**: RGBA (4 channels)
- **Resolution**: Same as the corresponding training image
- **RGB channels**: Always `(0, 0, 0)` - pure black for all pixels
- **Alpha channel**: 
  - `255` = foreground (object to segment)
  - `0` = background
  - `1-254` = soft edges/transitions (optional)

## Step-by-Step Instructions

### 1. Open Your Image
- `File > Open` - select your training image

### 2. Enter Quick Mask Mode
- Press `Shift+Q` (or click the small dashed square icon in bottom-left corner)
- Your image will get a red overlay

### 3. Paint Your Foreground Selection
- Press `D` to reset colors (black foreground, white background)
- Press `X` to swap them (white foreground, black background)
- Select the **Paintbrush Tool** (`P` key)
- **Set brush hardness**: In Tool Options panel, set **Hardness to 90-100** for crisp edges
  - 100 = perfectly sharp edges (minimal soft pixels)
  - 90-95 = 1-2 pixels of soft transition (like example dataset)
  - Lower values = wider soft boundary
- Adjust brush size: `[` and `]` keys to decrease/increase
- **Paint over your foreground object** - the red overlay disappears where you paint

**Quick Mask Tips:**
- **White** = selected (foreground)
- **Red overlay** = not selected (background)
- Press `X` to swap between white and black if you need to correct mistakes
- **Hardness setting controls edge crispness** - higher = sharper edges (recommend 80)
- Zoom in/out with mouse wheel for detail work
- Toggle Quick Mask (`Shift+Q`) to preview selection as "marching ants"

### 4. Exit Quick Mask Mode
- Press `Shift+Q` again
- Your painted area becomes a selection (marching ants around it)

### 5. Create the Mask Layer
- `Layer > New Layer`
  - Name it "mask"
  - Fill with: **Transparency**
  - Click OK

### 6. Add Alpha Channel
- Make sure the new mask layer is selected
- `Layer > Transparency > Add Alpha Channel`

### 7. Fill Selection with Black
- Set foreground color to black (press `D`)
- `Edit > Fill with FG Color`
- This creates black pixels with full opacity (alpha=255) in the selected area
- The unselected area remains transparent (alpha=0)

### 8. Deselect
- `Select > None`

### 9. Delete Original Image Layer
- In the Layers panel, click on the original image layer
- Right-click > `Delete Layer`
- You should now see:
  - Black foreground (your object)
  - Transparent background (checkerboard pattern)

### 10. Export as PNG
- `File > Export As`
- Save as `label_XX.png` (matching your image filename)
- **Important**: Ensure PNG options are set to save the alpha channel
- Click Export

## Verifying Your Mask

After creating your mask, verify it has the correct format:

```bash
python -c "from PIL import Image; import numpy as np; img = Image.open('label_01.png'); arr = np.array(img); print(f'Mode: {img.mode}'); print(f'Size: {img.size}'); print(f'RGB all black: {(arr[:,:,:3] == 0).all()}'); print(f'Alpha unique values: {np.unique(arr[:,:,3])}')"
```

Should output:
- `Mode: RGBA`
- Size should match your original image
- `RGB all black: True`
- Alpha should have values including 0 (background) and 255 (foreground)

## Tips for Better Masks

- **Use Feather** (`Select > Feather`) with 1-2px before filling to create soft edges
- **Soft brushes** in Quick Mask mode create natural alpha transitions at object boundaries
- **Zoom in** (mouse wheel) for precise edge work
- **Save your work** as `.xcf` while working, export to PNG when done
- **Quick Mask is non-destructive** - you can toggle it on/off to check your work

## Troubleshooting

**Problem**: After filling, everything turns black and I can't see my selection
- **Solution**: You filled the original layer instead of a new transparent layer. Follow step 5 carefully - create a NEW layer first.

**Problem**: Exported PNG doesn't have transparency
- **Solution**: Make sure you deleted the original image layer and only have the mask layer. Check PNG export options to ensure alpha channel is saved.

**Problem**: Mask file is the wrong size
- **Solution**: Make sure you didn't resize anything. The mask must be exactly the same resolution as the original image.
