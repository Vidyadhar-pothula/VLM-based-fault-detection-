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
        # The number of patches from BLIP ViT-Base defaults to 16x16 = 256 for a 224x224 input
        # However, transformers might dynamically adjust based on inputs. 
        # localize_faults expects flattened patch embeddings.
        heatmap_np, boxes, num_defective_patches = localize_faults(
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
    
    # 4. Criticality Score
    st.header("Fault Analysis")
    
    # Estimate grid size from number of patches (usually 16 or 14 depending on ViT patch size and crop)
    grid_size = int(np.sqrt(patch_emb.shape[1]))
    
    c_score, severity = calculate_criticality(
        num_defective_patches=num_defective_patches,
        w_real=w_real,
        w_img=image.width,
        a_threshold=a_threshold,
        grid_size=grid_size
    )
    
    # Determine severity color
    if severity == "Minor":
        sev_color = "green"
    elif severity == "Moderate":
        sev_color = "orange"
    else:
        sev_color = "red"
        
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.metric(label="Criticality Score (C)", value=f"{c_score:.2f}")
    with col_c2:
        st.markdown(f"### Severity Level: <span style='color:{sev_color}'>{severity}</span>", unsafe_allow_html=True)
        
    if num_defective_patches == 0:
        st.success("No Fault Detected")
        
    # 5. Captioning
    st.subheader("Image Caption")
    with st.spinner("Generating descriptive caption..."):
        # Guide caption towards anomalies if possible
        text_prompt = "A photo of " 
        caption = blip_loader.generate_caption(image, text_prompt)
        st.info(caption)
        
    # 6. VQA
    st.subheader("Visual Question Answering (VQA)")
    question = st.text_input("Ask a question about the image:")
    if question:
        with st.spinner("Answering..."):
            answer = blip_loader.answer_question(image, question)
            st.success(f"**Answer:** {answer}")
