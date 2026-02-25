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

def validate_part_answer(answer, bbox, image_height, image_width):
    """
    Prevents BLIP hallucinations by validating the answer spatially.
    """
    x, y, w, h = bbox
    y_center = y + h / 2
    x_center = x + w / 2

    # Relative Position Calculation
    rel_h = "center"
    if x_center < image_width * 0.4: rel_h = "left"
    elif x_center > image_width * 0.6: rel_h = "right"
    
    rel_v = "center"
    if y_center < image_height * 0.4: rel_v = "top"
    elif y_center > image_height * 0.6: rel_v = "bottom"
    
    location_str = f"{rel_v}-{rel_h}" if rel_v != "center" or rel_h != "center" else "center"

    # Validation Layer
    if y_center > image_height * 0.6 and "hood" in answer.lower():
        return "rear trunk or bumper", location_str

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
        
        # 2. Localize Faults (Step 1 of Architecture)
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
        # Step 2: Crop Region
        x, y, w, h = primary_box
        cropped_region = image.crop((x, y, x + w, y + h))
        
        # Step 3: Ask BLIP about cropped region only
        with st.spinner("Identifying car part in damaged region (Cropped VQA)..."):
            part_answer_raw = blip_loader.answer_question(
                cropped_region, 
                "Identify the exact car part visible in this cropped damaged region.", 
                is_region=True
            )
            
            # Step 4: Spatial Validation Layer
            img_w, img_h = image.size
            part_answer, location_str = validate_part_answer(part_answer_raw, primary_box, img_h, img_w)
            
            st.write(f"**Detected Part:** {part_answer}")
            st.write(f"**Spatial Location:** {location_str} region (Box: {x}, {y}, {x+w}, {y+h})")
            
        with st.spinner("Generating localized caption on cropped region..."):
            caption = blip_loader.generate_caption(cropped_region, is_region=True)
            st.info(f"**Regional Caption:** {caption}")
            
        # 5. Criticality Scoring
        grid_size = int(np.sqrt(patch_emb.shape[1]))
        c_score, severity = calculate_criticality(
            num_defective_patches=num_defective_patches,
            w_real=w_real,
            w_img=image.width,
            a_threshold=a_threshold,
            grid_size=grid_size
        )
        
        # Confidence Filter
        if c_score < 5: 
             st.warning("⚠️ No visible structural damage detected with high confidence.")
        else:
            sev_color = "green" if severity == "Minor" else ("orange" if severity == "Moderate" else "red")
            st.metric(label="Criticality Score (C)", value=f"{c_score:.2f}")
            st.markdown(f"### Severity Level: <span style='color:{sev_color}'>{severity}</span>", unsafe_allow_html=True)

    else:
        st.success("No significant anomalies detected.")

    # 6. Global VQA (Optional)
    st.subheader("Global Visual Question Answering")
    question = st.text_input("Ask a general question about the image:")
    if question:
        with st.spinner("Answering..."):
            answer = blip_loader.answer_question(image, question)
            st.success(f"**Answer:** {answer}")
