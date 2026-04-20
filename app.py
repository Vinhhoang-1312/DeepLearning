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
from models_lib import (
    MiniUNet, ConvLSTMUNet, ViTUNet, SwinUNet,
    CNNClassifier, BiGRUClassifier, ViTClassifier, SwinClassifier
)

# --- CONFIG ---
st.set_page_config(page_title="RBC Analysis Dashboard", layout="wide", page_icon="🔬")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEG_BASE = "results/segmentation"
CLS_BASE = "results/classification"

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton>button {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE STATE ---
if 'input_image' not in st.session_state: st.session_state.input_image = None
if 'mask' not in st.session_state: st.session_state.mask = None
if 'overlay' not in st.session_state: st.session_state.overlay = None
if 'cls_df' not in st.session_state: st.session_state.cls_df = None

# --- MODELS ---
@st.cache_resource
def get_seg_model(choice):
    if choice == "CNN":
        model = MiniUNet(in_ch=3, out_ch=1)
        path = os.path.join(SEG_BASE, "CNN_UNet_best.pt")
    elif choice == "RNN (ConvLSTM)":
        model = ConvLSTMUNet(in_ch=3, out_ch=1)
        path = os.path.join(SEG_BASE, "RNN_UNet_best.pt")
    elif choice == "Transformer (ViT)":
        model = ViTUNet(in_ch=3, out_ch=1)
        path = os.path.join(SEG_BASE, "ViT_UNet_best.pt")
    else:
        model = SwinUNet(in_ch=3, out_ch=1)
        path = os.path.join(SEG_BASE, "Swin_Seg_best.pt")
    
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()

@st.cache_resource
def get_cls_model(choice):
    if choice == "CNN":
        model = CNNClassifier(n_classes=9)
        path = os.path.join(CLS_BASE, "CNN_Cls_best.pt")
    elif choice == "RNN (BiGRU)":
        model = BiGRUClassifier(n_classes=9)
        path = os.path.join(CLS_BASE, "RNN_Cls_best.pt")
    elif choice == "Transformer (ViT)":
        model = ViTClassifier(n_classes=9)
        path = os.path.join(CLS_BASE, "ViT_Cls_best.pt")
    else:
        model = SwinClassifier(n_classes=9)
        path = os.path.join(CLS_BASE, "Swin_Cls_best.pt")
    
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Upload Blood Smear", type=["png", "jpg", "jpeg"])
    
    st.divider()
    seg_choice = st.selectbox("Segmentation Model", ["CNN", "RNN (ConvLSTM)", "Transformer (ViT)", "Swin Transformer"], index=0)
    cls_choice = st.selectbox("Classification Model", ["CNN", "RNN (BiGRU)", "Transformer (ViT)", "Swin Transformer"], index=0)
    threshold = st.slider("Seg Mask Threshold", 0.0, 1.0, 0.5, 0.05)
    
    if st.button("🚀 Run Full Analysis", use_container_width=True, type="primary"):
        if uploaded_file:
            input_img = Image.open(uploaded_file).convert("RGB")
            st.session_state.input_image = np.array(input_img)
            
            # 1. Segmentation
            s_model = get_seg_model(seg_choice)
            # Increase resolution to 512 for better cell visibility
            seg_res = 512 if "Transformer" not in seg_choice else 256
            tf_seg = transforms.Compose([transforms.Resize((seg_res, seg_res)), transforms.ToTensor()])
            input_tensor = tf_seg(input_img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                out = torch.sigmoid(s_model(input_tensor)).cpu().squeeze().numpy()
            
            mask = (out > threshold).astype(np.uint8) * 255
            mask = cv2.resize(mask, (input_img.size[0], input_img.size[1]))
            st.session_state.mask = mask
            
            # 2. Overlay
            overlay = st.session_state.input_image.copy()
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
            st.session_state.overlay = overlay
            
            # 3. Classification
            c_model = get_cls_model(cls_choice)
            tf_cls = transforms.Compose([transforms.ToPILImage(), transforms.Resize((80, 80)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
            
            final_data = []
            ELSAFTY_CLASSES = ["Rounded RBCs", "Ovalocytes", "Fragmented", "2 Overlapping", "3 Overlapping", "Burr Cells", "Teardrops", "Angled Cells", "Borderline Oval"]
            
            for i, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                if w < 10 or h < 10: continue
                crop = st.session_state.input_image[max(0, y-10):y+h+10, max(0, x-10):x+w+10]
                if crop.size == 0: continue
                
                with torch.no_grad():
                    logits = c_model(tf_cls(crop).unsqueeze(0).to(DEVICE))
                    prob = torch.softmax(logits, dim=1)
                    score, idx = torch.max(prob, 1)
                
                final_data.append({
                    "ID": i+1,
                    "Type": ELSAFTY_CLASSES[idx.item()],
                    "Conf": f"{score.item():.2%}",
                    "Box": (x,y,w,h)
                })
            st.session_state.cls_df = pd.DataFrame(final_data) if final_data else None
            st.success("✅ Analysis Completed!")
        else:
            st.warning("Please upload an image first!")

    if st.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.input_image = None
        st.session_state.mask = None
        st.session_state.overlay = None
        st.session_state.cls_df = None
        st.rerun()

# --- MAIN PAGE ---
st.title("🔬 RBC Analysis Dashboard")

if st.session_state.input_image is not None:
    col1, col2, col3 = st.columns([1, 1, 1.2])
    
    with col1:
        st.subheader("🖼️ Original")
        st.image(st.session_state.input_image, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Mask Overlay")
        if st.session_state.overlay is not None:
            st.image(st.session_state.overlay, use_container_width=True)
        else:
            st.info("Run analysis to see mask.")
            
    with col3:
        st.subheader("🏷️ Classification")
        if st.session_state.cls_df is not None:
            # Metrics
            total = len(st.session_state.cls_df)
            counts = st.session_state.cls_df["Type"].value_counts()
            
            m1, m2 = st.columns(2)
            m1.metric("Total RBCs", total)
            m2.metric("Most Frequent", counts.index[0] if not counts.empty else "N/A")
            
            st.dataframe(st.session_state.cls_df, use_container_width=True, hide_index=True)
            
            # Chart
            fig = px.pie(st.session_state.cls_df, names="Type", title="Distribution of Cell Types", hole=0.3)
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=40), height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run analysis to see classification.")
else:
    st.info("👈 Please upload an image and click 'Run Full Analysis' in the sidebar to begin.")
    # Skeleton containers
    c1, c2, c3 = st.columns(3)
    c1.empty(); c2.empty(); c3.empty()
