import torch
import torch.nn.functional as F
import numpy as np
import cv2

def localize_faults(patch_embeddings: torch.Tensor, global_embedding: torch.Tensor, image_size: tuple, threshold_percentile: float = 90):
    """
    Localizes faults by computing the cosine distance between patch embeddings and the global embedding.
    image_size: (width, height) of the original image
    threshold_percentile: Top percentile to consider as anomalous (e.g., 90 means top 10% are anomalous)
    """
    # patch_embeddings: (1, num_patches, hidden_size)
    # global_embedding: (1, hidden_size)
    
    # Squeeze batch dimension
    patch_emb = patch_embeddings.squeeze(0) # (num_patches, hidden_size)
    global_emb = global_embedding.squeeze(0) # (hidden_size,)
    
    num_patches = patch_emb.shape[0]
    
    # Calculate grid size (assumes square grid of patches)
    grid_size = int(np.sqrt(num_patches))
    
    # Normalize embeddings
    patch_emb_norm = F.normalize(patch_emb, p=2, dim=1)
    global_emb_norm = F.normalize(global_emb, p=2, dim=0)
    
    # Compute cosine similarity
    # Shape: (num_patches,)
    cos_sim = torch.mv(patch_emb_norm, global_emb_norm)
    
    # Anomaly score: 1 - cosine_similarity
    anomaly_scores = 1.0 - cos_sim
    
    anomaly_scores_np = anomaly_scores.cpu().numpy()
    
    # Reshape scores to grid
    anomaly_map = anomaly_scores_np.reshape(grid_size, grid_size)
    
    # Normalize anomaly map to 0-255 for visualization
    map_min = anomaly_map.min()
    map_max = anomaly_map.max()
    if map_max > map_min:
        anomaly_map_norm = (anomaly_map - map_min) / (map_max - map_min)
    else:
        anomaly_map_norm = np.zeros_like(anomaly_map)
        
    anomaly_map_uint8 = (anomaly_map_norm * 255).astype(np.uint8)
    
    # Upsample to original image size
    img_w, img_h = image_size
    anomaly_map_resized = cv2.resize(anomaly_map_uint8, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    
    # Thresholding to find significant anomalies
    threshold_value = np.percentile(anomaly_map_resized, threshold_percentile)
    _, binary_map = cv2.threshold(anomaly_map_resized, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    total_defective_area_px = 0
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter very small boxes to reduce noise
        if w * h > 100: 
            bounding_boxes.append((x, y, w, h))
            total_defective_area_px += (w * h)
            
    # Calculate equivalent defective patches for criticality scoring
    patch_area_px = (img_w / grid_size) * (img_h / grid_size)
    num_defective_patches = total_defective_area_px / patch_area_px if patch_area_px > 0 else 0
    
    primary_box = None
    if bounding_boxes:
        # Pick the largest box as the primary target for cropping/analysis
        primary_box = max(bounding_boxes, key=lambda b: b[2] * b[3])
    
    return anomaly_map_resized, bounding_boxes, num_defective_patches, primary_box
