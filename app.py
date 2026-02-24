import streamlit as st
from PIL import Image
import torch
import numpy as np
import os
import sys

# Ensure modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.blip_loader import BLIPLoader
from models.localization import localize_faults
from models.criticality import calculate_criticality
from utils.visualization import overlay_heatmap, draw_bounding_boxes

st.set_page_config(page_title="VLFD - Vision Language Fault Detector", layout="wide")

# Initialize models
@st.cache_resource
def load_models():
    return BLIPLoader()

blip_loader = load_models()

st.title("Vision Language Fault Detector (VLFD)")
st.write("Zero-shot fault detection using BLIP, implementing attention-based patch embedding deviations.")

# Sidebar for configuration
st.sidebar.header("Configuration")
w_real = st.sidebar.number_input("Real Width of Component (cm)", min_value=1.0, value=10.0, step=0.1)
a_threshold = st.sidebar.number_input("Critical Area Threshold (cm²)", min_value=0.1, value=5.0, step=0.1)
threshold_percentile = st.sidebar.slider("Anomaly Threshold Percentile (Top X %)", min_value=50, max_value=99, value=90)

uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        
    with st.spinner("Analyzing image for anomalies..."):
        # 1. Extract Embeddings
        patch_emb, global_emb = blip_loader.get_image_embeddings(image)
        
        # 2. Localize Faults
        heatmap_np, boxes, num_defective_patches, primary_box = localize_faults(
            patch_emb, 
            global_emb, 
            image.size, 
            threshold_percentile=threshold_percentile
        )
        
        # 3. Create Visualizations
        heatmap_img = overlay_heatmap(image, heatmap_np)
        bbox_img = draw_bounding_boxes(image, boxes)
        
    with col2:
        st.subheader("Anomaly Heatmap")
        st.image(heatmap_img, use_column_width=True)
        
    with col3:
        st.subheader("Fault Localization (BBox)")
        st.image(bbox_img, use_column_width=True)
        
    st.markdown("---")
    
    # 4. Regional Analysis & Hallucination Mitigation
    st.header("Localized Fault Analysis")
    
    if primary_box:
        x, y, w, h = primary_box
        # Crop the region (PIL uses left, top, right, bottom)
        cropped_region = image.crop((x, y, x + w, y + h))
        
        # Spatial Heuristics
        img_w, img_h = image.size
        center_x, center_y = x + w/2, y + h/2
        
        rel_h = "center"
        if center_x < img_w * 0.4: rel_h = "left"
        elif center_x > img_w * 0.6: rel_h = "right"
        
        rel_v = "center"
        if center_y < img_h * 0.4: rel_v = "top"
        elif center_y > img_h * 0.6: rel_v = "bottom"
        
        location_str = f"{rel_v}-{rel_h}" if rel_v != "center" or rel_h != "center" else "center"
        
        # Constrained reasoning on crop
        with st.spinner("Identifying car part in damaged region..."):
            part_answer = blip_loader.answer_question(cropped_region, "What car part is this?", is_region=True)
            
            # Rule-based validation layer
            # Example: If bbox is bottom half, it shouldn't be hood
            if center_y > img_h * 0.6 and "hood" in part_answer.lower():
                part_answer = "lower body/bumper (Corrected from hallucinated 'hood')"
            
            st.write(f"**Detected Part:** {part_answer}")
            st.write(f"**Spatial Location:** {location_str} region (Box: {x}, {y}, {x+w}, {y+h})")
            
        with st.spinner("Generating localized caption..."):
            caption = blip_loader.generate_caption(cropped_region, is_region=True)
            st.info(f"**Regional Caption:** {caption}")
            
        # Confidence Filter
        grid_size = int(np.sqrt(patch_emb.shape[1]))
        c_score, severity = calculate_criticality(
            num_defective_patches=num_defective_patches,
            w_real=w_real,
            w_img=image.width,
            a_threshold=a_threshold,
            grid_size=grid_size
        )
        
        # Final Verification Rule
        if c_score < 5: # Low confidence threshold
             st.warning("⚠️ No visible structural damage detected with high confidence.")
        else:
            # Determine severity color
            sev_color = "green" if severity == "Minor" else ("orange" if severity == "Moderate" else "red")
            col_c1, col_c2 = st.columns(2)
            with col_c1: st.metric(label="Criticality Score (C)", value=f"{c_score:.2f}")
            with col_c2: st.markdown(f"### Severity Level: <span style='color:{sev_color}'>{severity}</span>", unsafe_allow_html=True)

    else:
        st.success("No significant anomalies detected.")

    # 6. Global VQA (Optional)
    st.subheader("Global Visual Question Answering")
    question = st.text_input("Ask a general question about the image:")
    if question:
        with st.spinner("Answering..."):
            answer = blip_loader.answer_question(image, question)
            st.success(f"**Answer:** {answer}")
