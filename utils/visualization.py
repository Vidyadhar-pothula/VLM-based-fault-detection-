import cv2
import numpy as np
from PIL import Image

def overlay_heatmap(image: Image.Image, heatmap_np: np.ndarray, alpha: float = 0.5):
    """
    Overlays a heat map on the original image.
    image: Original PIL Image
    heatmap_np: 2D numpy array containing heatmap values 0-255
    """
    img_np = np.array(image)
    if img_np.shape[2] == 4: # Handle RGBA
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)
    
    # Overlay heatmap on original image
    overlaid_img = cv2.addWeighted(img_np, 1 - alpha, heatmap_colored, alpha, 0)
    
    return Image.fromarray(overlaid_img)

def draw_bounding_boxes(image: Image.Image, boxes: list):
    """
    Draws bounding boxes around anomalies on the image.
    boxes: list of tuples (x, y, w, h)
    """
    img_np = np.array(image)
    if len(img_np.shape) == 2: # Handle grayscale
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4: # Handle RGBA
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
    for (x, y, w, h) in boxes:
        # Draw red bounding box with thickness 2
        cv2.rectangle(img_np, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
    return Image.fromarray(img_np)
