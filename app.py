import streamlit as st
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
from models_lib import MiniUNet, CNNClassifier, BiGRUClassifier, ViTClassifier

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEG_BASE = "results/segmentation"
CLS_BASE = "results/classification"

# Page Config
st.set_page_config(page_title="RBC Analytics Pro", layout="wide", page_icon="🔬")

# Sidebar - Settings
with st.sidebar:
    st.title("🔬 Settings")
    conf_threshold = st.slider("Segmentation Threshold", 0.0, 1.0, 0.5, 0.05)
    seg_model_choice = st.selectbox("Segmentation Model", ["CNN", "RNN (BiGRU)", "Transformer (ViT)"])
    model_choice = st.selectbox("Classification Model", ["CNN", "RNN (BiGRU)", "Transformer (ViT)"])
    if st.button("🗑️ Clear All Session Data", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- State Initialization ---
if "input_image" not in st.session_state:
    st.session_state.input_image = None
if "mask" not in st.session_state:
    st.session_state.mask = None
if "crops" not in st.session_state:
    st.session_state.crops = []
if "cls_results" not in st.session_state:
    st.session_state.cls_results = None

# --- Helper Functions ---
@st.cache_resource
def get_seg_model(choice):
    model = MiniUNet(in_ch=3, out_ch=1)
    if choice == "CNN":
        path = os.path.join(SEG_BASE, "CNN_UNet_best.pt")
    elif choice == "RNN (BiGRU)":
        path = os.path.join(SEG_BASE, "RNN_UNet_best.pt")
    else:
        path = os.path.join(SEG_BASE, "ViT_UNet_best.pt")
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        st.sidebar.success(f"Seg model loaded: {os.path.basename(path)}")
    else:
        st.sidebar.error(f"Seg weights not found: {path}")
    return model.to(DEVICE).eval()

def load_cls_model(choice):
    try:
        if choice == "CNN":
            model = CNNClassifier(n_classes=9)
            path = os.path.join(CLS_BASE, "CNN_Cls_best.pt")
        elif choice == "RNN (BiGRU)":
            model = BiGRUClassifier(n_classes=9)
            path = os.path.join(CLS_BASE, "RNN_Cls_best.pt")
        else:
            model = ViTClassifier(n_classes=9)
            path = os.path.join(CLS_BASE, "ViT_Cls_best.pt")
        
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.to(DEVICE).eval()
            return model
        else:
            st.error(f"Weights not found at {path}")
            return None
    except Exception as e:
        st.error(f"Error loading {choice} model: {e}")
        st.info("Tip: If you see a 'size mismatch', try restarting Streamlit to clear cached modules.")
        return None

# --- Main UI ---
st.title("🔬 RBC Analysis Dashboard")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📁 Upload & Preview", "🎯 Segmentation Analysis", "🏷️ Classification Specs"])

# --- TAB 1: UPLOAD ---
with tab1:
    uploaded_file = st.file_uploader("Choose a blood smear image...", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.session_state.input_image = np.array(img)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(st.session_state.input_image, caption="Uploaded Blood Smear", use_container_width=True)
        with col2:
            st.success("Image uploaded successfully!")
            if st.button("🚀 Start Pipeline", use_container_width=True):
                # Trigger Segmentation
                seg_model = get_seg_model(seg_model_choice)
                tf_seg = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
                input_tensor = tf_seg(img).unsqueeze(0).to(DEVICE)
                
                with st.spinner("Segmenting cells..."):
                    with torch.no_grad():
                        out = torch.sigmoid(seg_model(input_tensor)).cpu().squeeze().numpy()
                    mask = (out > conf_threshold).astype(np.uint8) * 255
                    st.session_state.mask = cv2.resize(mask, (img.size[0], img.size[1]))
                    st.success("Segmentation Done! Move to Tab 2.")
                    st.session_state.cls_results = None # Reset old results

# --- TAB 2: SEGMENTATION ---
with tab2:
    if st.session_state.mask is not None:
        st.header("🎯 Segmentation Results")
        mask_binary = st.session_state.mask
        orig = st.session_state.input_image
        
        # Overlay
        overlay = orig.copy()
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        
        col1, col2 = st.columns(2)
        col1.image(mask_binary, caption="Binary Mask", use_container_width=True)
        col2.image(overlay, caption="Detected Contours", use_container_width=True)
        
        # Metrics
        st.metric("Total Cells Detected", len(contours))
        
        # Extract Crops for next step
        crops = []
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            # Add padding
            pad = 10
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(orig.shape[1], x+w+pad), min(orig.shape[0], y+h+pad)
            crop = orig[y1:y2, x1:x2]
            crops.append(crop)
        st.session_state.crops = crops
    else:
        st.info("Please upload an image and run the pipeline in Tab 1.")

# --- TAB 3: CLASSIFICATION ---
with tab3:
    if not st.session_state.crops:
        st.info("Please complete segmentation in Tab 2 first.")
    else:
        st.header(f"🏷️ Classification ({model_choice})")
        
        if st.button("🔍 Run Classification", use_container_width=True):
            cls_model = load_cls_model(model_choice)
            if cls_model:
                tf_cls = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((80, 80)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5]*3, [0.5]*3)
                ])
                
                results = []
                # Real labels from Elsafty dataset
                ELSAFTY_CLASSES = [
                    "Rounded RBCs", "Ovalocytes", "Fragmented", "2 Overlapping",
                    "3 Overlapping", "Burr Cells", "Teardrops", "Angled Cells", "Borderline Oval"
                ]

                progress_bar = st.progress(0)
                for i, crop in enumerate(st.session_state.crops):
                    img_tensor = tf_cls(crop).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        logits = cls_model(img_tensor)
                        probs = torch.softmax(logits, dim=1)
                        conf, pred = torch.max(probs, 1)
                    
                    results.append({
                        "Cell ID": i + 1,
                        "Prediction": ELSAFTY_CLASSES[pred.item()],
                        "Confidence": conf.item(),
                        "Image": crop
                    })
                    progress_bar.progress((i + 1) / len(st.session_state.crops))
                
                st.session_state.cls_results = results

        if st.session_state.cls_results:
            df = pd.DataFrame(st.session_state.cls_results)
            
            # Summary Chart
            st.subheader("📊 Distribution Analysis")
            fig = px.bar(df["Prediction"].value_counts().reset_index(), 
                         x="Prediction", y="count", color="Prediction",
                         title="Detected Cell Types")
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Table
            st.subheader("📋 Detailed Records")
            st.dataframe(df[["Cell ID", "Prediction", "Confidence"]], use_container_width=True)
            
            # Visual Gallery
            st.subheader("🖼️ Cell Gallery")
            rows = st.columns(4)
            for i, res in enumerate(st.session_state.cls_results[:12]): # Show first 12
                with rows[i % 4]:
                    st.image(res["Image"], caption=f"ID:{res['Cell ID']} - {res['Prediction']}", use_container_width=True)
