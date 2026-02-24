# Vision Language Fault Detector (VLFD)

VLFD is a zero-shot fault detection system built using the **BLIP** (Bootstrapping Language-Image Pre-training) framework. It leverages BLIP's internal ViT backbone and cross-modal reasoning to identify, localize, and analyze defects in industrial components without any specific training on anomaly datasets.

## 🚀 Features
- **Zero-Shot Fault Localization**: Uses patch-level embedding deviations from BLIP's vision encoder to generate anomaly heatmaps and bounding boxes.
- **Criticality Scoring**: Implements a dedicated formula to calculate the severity of detected faults (Minor, Moderate, Critical) based on pixel-to-cm conversion.
- **Automated Captioning**: Generates descriptive captions reflecting the detected anomalies.
- **VQA Interface**: Integrated Visual Question Answering allows users to ask specific questions about the component's state.
- **Fast Startup**: Optimized with a local model weight loader.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Vidyadhar-pothula/VLM-based-fault-detection- .
   ```

2. **Set up Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Models**:
   Run the download script to cache weights locally for faster startup:
   ```bash
   python download_models.py
   ```

## 💻 Usage
Run the Streamlit application:
```bash
streamlit run app.py
```

## 🔬 Core Methodology
- **Patch-level Localization**: Extracts embeddings from the last ViT layer of BLIP and computes cosine similarity against the global image embedding to identify anomalous regions.
- **Deviation Scoring**: High cosine distance candidates are thresholded and clustered into bounding boxes.
- **Severity Classification**:
  - $C < 30 \rightarrow$ Minor
  - $30 \le C < 70 \rightarrow$ Moderate
  - $C \ge 70 \rightarrow$ Critical

## 📄 Requirements
- PyTorch
- Transformers (HuggingFace)
- Streamlit
- OpenCV
- Numpy
- Pillow

---
*Developed as a Rebuild of the VLFD Paper.*
