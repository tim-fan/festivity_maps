

# based on https://visionbrick.com/introduction-to-dinov3-generating-similarity-maps-with-vision-transformers/
import torch
import torchvision.transforms as T
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l



# CHECKPOINT_PATH = "weights/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
# MODEL_NAME = "dinov3_vitl16_dinotxt_tet1280d20h24l"

# # dinov3_vitl16_dinotxt_tet1280d20h24l, tokenizer = torch.hub.load(DINOV3_LOCATION, MODEL_NAME, source='local', weights=CHECKPOINT_PATH, backbone_weights=CHECKPOINT_PATH)

# model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(
#     weights="weights/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
#     backbone_weights="weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
# )




def main():
    # DINOv3 setup
    

    DINOV3_LOCATION = "dinov3"
    CHECKPOINT_PATH = "weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    MODEL_NAME = "dinov3_vits16"    
    model = torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=MODEL_NAME,
        source="local",
        weights=CHECKPOINT_PATH,
    )

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

    model.eval().cuda()

    img_path = "../datasets/20241204_low_exposure/images/IMG_20241204_185634.jpg"
    orig_bgr = cv2.imread(img_path)
    img = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    
    # Use 320x320 which is exactly 20x20 patches of 16x16
    RESIZE = 320
    PATCH_SIZE = 16
    img_resized = cv2.resize(img, (RESIZE, RESIZE), interpolation=cv2.INTER_AREA)
    
    plt.imshow(img_resized)


    # Preprocess for DINOv3 
    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # add batch dimension and apply transforms
    inp = transform(img_resized).unsqueeze(0).cuda()
    
    print(f"Input shape: {inp.shape}")

    # Extract patch embeddings (features)
    with torch.no_grad():
        features = model.forward_features(inp)
        features = features['x_norm_patchtokens'][0].cpu().numpy()  # 'x_norm_patchtokens': contains normalized features for each patch token.
        print("size of the features:", features.shape) 

    num_patches = features.shape[0]
    grid_size = int(np.sqrt(num_patches))
    
    # Calculate actual patch size on display
    actual_patch_size = RESIZE // grid_size
    
    # Draw EXACT patch grid
    grid_img = img_resized.copy()
    for i in range(1, grid_size):
        x = i * actual_patch_size
        cv2.line(grid_img, (x, 0), (x, RESIZE), (255, 0, 0), 2)
    for j in range(1, grid_size):
        y = j * actual_patch_size
        cv2.line(grid_img, (0, y), (RESIZE, y), (255, 0, 0), 2)
    
    # Click handler
    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        
        # Calculate which patch was clicked
        patch_x = min(x // actual_patch_size, grid_size - 1)
        patch_y = min(y // actual_patch_size, grid_size - 1)
        idx = patch_y * grid_size + patch_x
        
        print(f"Clicked patch ({patch_y}, {patch_x}), index: {idx}")
        
        # Get reference feature
        # feats --> 400,384
        referance_feature = features[idx] # 1,384
        
        """ 
        Compute cosine similarity with all patches
        divide dot product by product of norms(l2) to normalize
        @ --> dot product
        np.linalg.norm --> l2 norm 
        """
        print(features.shape) # 400,384
        print(referance_feature.shape) # 384,1
        similarities = features @ referance_feature / (np.linalg.norm(features, axis=1) * np.linalg.norm(referance_feature) + 1e-8) # 400,1
        similarities = similarities.reshape(grid_size, grid_size)  # 20,20
    
        # Resize similarity map to match image size
        sim_resized = cv2.resize(similarities, (RESIZE, RESIZE), interpolation=cv2.INTER_CUBIC)
        sim_norm = cv2.normalize(sim_resized, None, 0, 255, cv2.NORM_MINMAX)
        sim_color = cv2.applyColorMap(sim_norm.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        
        # Mark the clicked patch with a rectangle
        marked_img = sim_color.copy()
        top_left = (patch_x * actual_patch_size, patch_y * actual_patch_size)
        bottom_right = ((patch_x + 1) * actual_patch_size, (patch_y + 1) * actual_patch_size)
        
        # draw point to target patch
        center = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)
        cv2.circle(marked_img, center, radius=5, color=(0, 0, 255), thickness=-1)
        cv2.imshow("Cosine Similarity Map", marked_img)
    
    # Display
    cv2.namedWindow("DINOv3 Patches (click one)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("DINOv3 Patches (click one)", on_click)
    cv2.imshow("DINOv3 Patches (click one)", grid_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
