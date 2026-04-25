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

ELSAFTY_CLASSES = [
    "Rounded RBCs", "Ovalocytes", "Fragmented",
    "2 Overlapping", "3 Overlapping", "Burr Cells",
    "Teardrops", "Angled Cells", "Borderline Oval"
]

CLASS_COLORS = {
    "Rounded RBCs": "#2ecc71", "Ovalocytes": "#f39c12", "Fragmented": "#e74c3c",
    "2 Overlapping": "#95a5a6", "3 Overlapping": "#7f8c8d", "Burr Cells": "#3498db",
    "Teardrops": "#e67e22", "Angled Cells": "#9b59b6", "Borderline Oval": "#1abc9c"
}

CLINICAL_ADVICE = {
    "Rounded RBCs":   {"status": "Khỏe mạnh",    "icon": "✅", "msg": "Tế bào hồng cầu hình tròn đều bình thường.",                     "advice": "Duy trì chế độ ăn uống và sinh hoạt lành mạnh!",                                   "severity": 0},
    "Ovalocytes":     {"status": "Cần chú ý",     "icon": "⚠️", "msg": "Phát hiện tế bào Ovalocytes (hình bầu dục).",                    "advice": "Có thể thiếu Vitamin B12 hoặc Sắt. Nên bổ sung thực phẩm giàu sắt hoặc tham vấn bác sĩ.", "severity": 2},
    "Teardrops":      {"status": "Cảnh báo",      "icon": "🔶", "msg": "Phát hiện tế bào hình giọt nước (Teardrops).",                   "advice": "Thường liên quan đến vấn đề tủy xương/thiếu máu nặng. Xét nghiệm máu tổng quát.",  "severity": 3},
    "Fragmented":     {"status": "Nguy cơ cao",   "icon": "🚨", "msg": "Cảnh báo: Phát hiện mảnh vỡ hồng cầu (Fragmented).",            "advice": "Dấu hiệu thiếu máu tán huyết. Cần đến bác sĩ chuyên khoa huyết học ngay.",          "severity": 4},
    "Burr Cells":     {"status": "Theo dõi",       "icon": "🔵", "msg": "Phát hiện tế bào Burr (gai nhỏ).",                               "advice": "Có thể do yếu tố thận hoặc lỗi chuẩn bị tiêu bản. Kiểm tra chức năng thận nếu kéo dài.", "severity": 1},
    "2 Overlapping":  {"status": "Ghi chú",        "icon": "📌", "msg": "Tế bào (Ảnh) đang nằm chồng lên nhau.",                          "advice": "Mật độ tế bào cao / ảnh cắt chưa chuẩn, ảnh hưởng độ chính xác.",                   "severity": 0},
    "3 Overlapping":  {"status": "Ghi chú",        "icon": "📌", "msg": "Nhiều tế bào chồng lấp phức tạp.",                               "advice": "Thử upload ảnh thưa hơn/cắt chính xác hơn.",                                        "severity": 0},
    "Angled Cells":   {"status": "Theo dõi",       "icon": "🔵", "msg": "Phát hiện tế bào có góc cạnh.",                                  "advice": "Thường không có ý nghĩa lâm sàng nghiêm trọng.",                                    "severity": 1},
    "Borderline Oval":{"status": "Theo dõi",       "icon": "🔵", "msg": "Tế bào hơi dài.",                                                "advice": "Biến thể nhẹ của hồng cầu bình thường.",                                            "severity": 1}
}

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, .stApp { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white; padding: 30px 40px; border-radius: 16px;
        margin-bottom: 24px; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p { margin: 4px 0 0; opacity: 0.8; font-size: 0.95rem; }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.1); border-radius: 12px;
        padding: 20px; text-align: center; color: #e0e0e0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        transition: transform 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #4fc3f7; }
    .metric-card .label { font-size: 0.85rem; opacity: 0.7; margin-top: 4px; }
    
    .insight-card {
        border-radius: 12px; padding: 20px 24px; margin: 8px 0;
        border-left: 4px solid; backdrop-filter: blur(10px);
    }
    .insight-ok    { background: rgba(46,204,113,0.1);  border-color: #2ecc71; }
    .insight-watch { background: rgba(52,152,219,0.1);  border-color: #3498db; }
    .insight-warn  { background: rgba(243,156,18,0.1);  border-color: #f39c12; }
    .insight-danger{ background: rgba(231,76,60,0.1);   border-color: #e74c3c; }
    
    .crop-gallery {
        display: flex; flex-wrap: wrap; gap: 8px;
        max-height: 400px; overflow-y: auto;
        padding: 8px; background: rgba(0,0,0,0.05); border-radius: 8px;
    }
    
    .mode-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600;
    }
    .mode-full  { background: #3498db; color: white; }
    .mode-single{ background: #9b59b6; color: white; }
    
    .stButton>button {
        border-radius: 8px; font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE STATE ---
for key in ['input_image', 'mask', 'overlay', 'cls_df', 'crops', 'crop_labels']:
    if key not in st.session_state:
        st.session_state[key] = None
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "Full Slide"

# ─────────────────────────────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_seg_model(choice):
    models_map = {
        "CNN":              (MiniUNet,     "CNN_UNet_best.pt"),
        "RNN (ConvLSTM)":   (ConvLSTMUNet, "RNN_UNet_best.pt"),
        "Transformer (ViT)":(ViTUNet,      "ViT_UNet_best.pt"),
        "Swin Transformer": (SwinUNet,     "Swin_Seg_best.pt"),
    }
    cls, fname = models_map[choice]
    model = cls(in_ch=3, out_ch=1)
    path = os.path.join(SEG_BASE, fname)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    else:
        st.warning(f"⚠️ Weights not found: `{path}` — using random weights")
    return model.to(DEVICE).eval()

@st.cache_resource
def get_cls_model(choice):
    models_map = {
        "CNN":              (CNNClassifier,  "CNN_Cls_best.pt"),
        "RNN (BiGRU)":      (BiGRUClassifier,"RNN_Cls_best.pt"),
        "Transformer (ViT)":(ViTClassifier,  "ViT_Cls_best.pt"),
        "Swin Transformer": (SwinClassifier, "Swin_Cls_best.pt"),
    }
    cls, fname = models_map[choice]
    model = cls(n_classes=9)
    path = os.path.join(CLS_BASE, fname)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    else:
        st.warning(f"⚠️ Weights not found: `{path}` — using random weights")
    return model.to(DEVICE).eval()

# ─────────────────────────────────────────────────────────────────
# PREPROCESSING (matches training pipeline)
# ─────────────────────────────────────────────────────────────────
TF_CLS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((80, 80)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def get_seg_transform(choice):
    seg_res = 256 if "Transformer" in choice or "Swin" in choice else 512
    return transforms.Compose([
        transforms.Resize((seg_res, seg_res)),
        transforms.ToTensor()
    ]), seg_res

# ─────────────────────────────────────────────────────────────────
# CORE PIPELINE FUNCTIONS
# ─────────────────────────────────────────────────────────────────
def run_segmentation(pil_img, seg_choice, threshold):
    """Run segmentation model on a full slide image → returns binary mask."""
    s_model = get_seg_model(seg_choice)
    tf_seg, seg_res = get_seg_transform(seg_choice)
    input_tensor = tf_seg(pil_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        out = torch.sigmoid(s_model(input_tensor)).cpu().squeeze().numpy()
    
    mask = (out > threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask, (pil_img.size[0], pil_img.size[1]))
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return mask


def extract_cells(image_np, mask):
    """Extract individual cell crops from segmented mask using watershed-inspired approach."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_h, img_w = image_np.shape[:2]
    min_cell_area = max(100, (img_h * img_w) * 0.0005)   # adaptive minimum
    max_cell_area = (img_h * img_w) * 0.3                 # no cell > 30% of image
    
    crops = []
    boxes = []
    valid_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_cell_area or area > max_cell_area:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        if aspect_ratio > 5:  # skip extreme elongated noise
            continue
        
        pad = max(3, int(min(w, h) * 0.1))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img_w, x + w + pad), min(img_h, y + h + pad)
        
        crop = image_np[y1:y2, x1:x2]
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            continue
            
        crops.append(crop)
        boxes.append((x, y, w, h))
        valid_contours.append(cnt)
    
    return crops, boxes, valid_contours


def classify_cells(crops, cls_choice):
    """Classify a list of cell crops → returns list of (class_name, confidence)."""
    c_model = get_cls_model(cls_choice)
    results = []
    
    for crop in crops:
        with torch.no_grad():
            tensor = TF_CLS(crop).unsqueeze(0).to(DEVICE)
            logits = c_model(tensor)
            prob = torch.softmax(logits, dim=1)
            score, idx = torch.max(prob, 1)
            results.append((ELSAFTY_CLASSES[idx.item()], score.item()))
    
    return results


def classify_single_cell(image_np, cls_choice):
    """Classify a single cell image directly."""
    c_model = get_cls_model(cls_choice)
    with torch.no_grad():
        tensor = TF_CLS(image_np).unsqueeze(0).to(DEVICE)
        logits = c_model(tensor)
        prob = torch.softmax(logits, dim=1)
        
        # Get top-3 predictions
        top3_scores, top3_idx = torch.topk(prob, 3, dim=1)
        results = []
        for i in range(3):
            results.append((
                ELSAFTY_CLASSES[top3_idx[0, i].item()],
                top3_scores[0, i].item()
            ))
    return results

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    app_mode = st.radio(
        "Operating Mode",
        ["🔬 Phân tích Toàn cảnh (Full Slide)", "🧬 Phân loại 1 Tế bào (Single Cell)"],
        help="Full Slide: Segmentation → tách từng tế bào → phân loại.\nSingle Cell: upload 1 ảnh tế bào đã crop → phân loại trực tiếp."
    )
    is_full_slide = "Full Slide" in app_mode
    
    # Reset state if mode changes
    if app_mode != st.session_state.app_mode:
        for key in ['input_image', 'mask', 'overlay', 'cls_df', 'crops', 'crop_labels']:
            st.session_state[key] = None
        st.session_state.app_mode = app_mode
        st.rerun()

    st.divider()
    uploaded_file = st.file_uploader(
        "Upload Image", type=["png", "jpg", "jpeg"],
        help="Full Slide: ảnh tiêu bản toàn cảnh.\nSingle Cell: ảnh 1 tế bào đã crop sẵn."
    )

    if is_full_slide:
        model_family = st.selectbox("AI Architecture Pipeline", ["CNN", "RNN", "Transformer (ViT)", "Swin Transformer"])
        seg_map = {"CNN": "CNN", "RNN": "RNN (ConvLSTM)", "Transformer (ViT)": "Transformer (ViT)", "Swin Transformer": "Swin Transformer"}
        cls_map = {"CNN": "CNN", "RNN": "RNN (BiGRU)", "Transformer (ViT)": "Transformer (ViT)", "Swin Transformer": "Swin Transformer"}
        seg_choice = seg_map[model_family]
        cls_choice = cls_map[model_family]
        
        threshold = st.slider("Seg Mask Threshold", 0.0, 1.0, 0.5, 0.05)
        conf_threshold = st.slider("Min Classification Confidence", 0.0, 1.0, 0.3, 0.05,
                                   help="Cells dưới ngưỡng này sẽ được đánh dấu 'Uncertain'")
        run_btn = st.button("🚀 Run Full Pipeline", use_container_width=True, type="primary")
    else:
        model_family = st.selectbox("AI Architecture Pipeline", ["CNN", "RNN", "Transformer (ViT)", "Swin Transformer"])
        cls_map = {"CNN": "CNN", "RNN": "RNN (BiGRU)", "Transformer (ViT)": "Transformer (ViT)", "Swin Transformer": "Swin Transformer"}
        cls_choice = cls_map[model_family]
        run_btn = st.button("🚀 Classify Single Cell", use_container_width=True, type="primary")
    
    st.divider()
    if st.button("🗑️ Clear All", use_container_width=True):
        for key in ['input_image', 'mask', 'overlay', 'cls_df', 'crops', 'crop_labels']:
            st.session_state[key] = None
        st.rerun()

# ─────────────────────────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────────────────────────
if run_btn:
    if not uploaded_file:
        st.warning("⚠️ Please upload an image first!")
    else:
        input_img = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(input_img)
        st.session_state.input_image = image_np
        
        w, h = input_img.size
        
        if is_full_slide:
            # --- VALIDATION: check if image is too small for full slide ---
            if w < 150 and h < 150:
                st.error("🚫 LỖI CẤU HÌNH CỦA BẠN: Ảnh này quá nhỏ (chỉ chứa 1 tế bào). Bạn đang upload ảnh từ thư mục **`test_classification`** vào chế độ Full Slide (Tiêu bản rộng)!\n\n👉 Hãy nhìn sang Menu bên trái, chọn lại chế độ **🧬 Phân loại 1 Tế bào (Single Cell)** để chạy ảnh này.")
            else:
                with st.spinner("🔄 Step 1/3: Running Segmentation..."):
                    mask = run_segmentation(input_img, seg_choice, threshold)
                    st.session_state.mask = mask
                
                with st.spinner("🔄 Step 2/3: Extracting Cells..."):
                    crops, boxes, valid_contours = extract_cells(image_np, mask)
                    st.session_state.crops = crops
                    
                    # Draw overlay
                    overlay = image_np.copy()
                    cv2.drawContours(overlay, valid_contours, -1, (0, 255, 0), 2)
                    for i, (x, y, w, h) in enumerate(boxes):
                        cv2.putText(overlay, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    st.session_state.overlay = overlay
                
                if not crops:
                    st.warning("⚠️ Không phát hiện được tế bào nào! Thử điều chỉnh Seg Mask Threshold.")
                    st.session_state.cls_df = None
                else:
                    with st.spinner(f"🔄 Step 3/3: Classifying {len(crops)} cells..."):
                        cls_results = classify_cells(crops, cls_choice)
                        
                        final_data = []
                        for i, ((cls_name, conf), (x, y, w, h)) in enumerate(zip(cls_results, boxes)):
                            label = cls_name if conf >= conf_threshold else f"❓ Uncertain ({cls_name})"
                            final_data.append({
                                "ID": i+1,
                                "Type": label,
                                "Raw Type": cls_name,
                                "Confidence": f"{conf:.1%}",
                                "Conf_val": conf,
                                "Box": f"({x},{y},{w},{h})"
                            })
                        
                        st.session_state.cls_df = pd.DataFrame(final_data)
                        st.session_state.crop_labels = cls_results
                    
                    st.success(f"✅ Analysis Complete! Found {len(crops)} cells.")
        
        else:
            # --- SINGLE CELL MODE ---
            # --- VALIDATION: check if image is too big for a single cell ---
            if w > 150 or h > 150:
                st.error("🚫 LỖI CẤU HÌNH CỦA BẠN: Ảnh này quá lớn (chứa hàng tá tế bào). Bạn đang upload ảnh từ thư mục **`test_segmentation`** vào chế độ Single Cell (1 Tế bào)!\n\n👉 Hãy nhìn sang Menu bên trái, chọn lại chế độ **🔬 Phân tích Toàn cảnh (Full Slide)** để máy chạy bóc tách từng tế bào ra trước nhé.")
            else:
                st.session_state.mask = None
                st.session_state.overlay = None
                st.session_state.crops = None
                
                with st.spinner("🔄 Classifying..."):
                    top3 = classify_single_cell(image_np, cls_choice)
                    
                    st.session_state.cls_df = pd.DataFrame([{
                        "Rank": i+1,
                        "Type": name,
                        "Confidence": f"{conf:.1%}",
                        "Conf_val": conf
                    } for i, (name, conf) in enumerate(top3)])
                
                st.success("✅ Classification Complete!")

# ─────────────────────────────────────────────────────────────────
# MAIN DISPLAY
# ─────────────────────────────────────────────────────────────────
mode_label = "Toàn Cảnh (Full Slide)" if is_full_slide else "1 Tế bào (Single Cell)"
mode_css = "mode-full" if is_full_slide else "mode-single"
st.markdown(f"""
<div class="main-header">
    <h1>🔬 RBC Analysis Dashboard</h1>
    <p>Red Blood Cell Morphology Analysis &nbsp; <span class="mode-badge {mode_css}">{mode_label}</span></p>
</div>
""", unsafe_allow_html=True)

if st.session_state.input_image is not None:
    
    if is_full_slide:
        # ═══════════════════════════════════════════════════════
        # FULL SLIDE LAYOUT
        # ═══════════════════════════════════════════════════════
        tab_main, tab_crops = st.tabs(["📊 Results", "🔍 Crop Inspector"])
        
        with tab_main:
            col1, col2 = st.columns([1.2, 1])
            
            with col1:
                img_tab1, img_tab2, img_tab3 = st.tabs(["🖼️ Original", "🎭 Segmentation Mask", "🎯 Overlay"])
                with img_tab1:
                    st.image(st.session_state.input_image, use_container_width=True)
                with img_tab2:
                    if st.session_state.mask is not None:
                        st.image(st.session_state.mask, use_container_width=True, clamp=True)
                    else:
                        st.info("Run pipeline to see segmentation mask")
                with img_tab3:
                    if st.session_state.overlay is not None:
                        st.image(st.session_state.overlay, use_container_width=True)
                    else:
                        st.info("Run pipeline to see overlay")
            
            with col2:
                st.markdown("### 📋 Classification Results")
                if st.session_state.cls_df is not None and not st.session_state.cls_df.empty:
                    df = st.session_state.cls_df
                    
                    # Metrics row
                    total = len(df)
                    high_conf = len(df[df['Conf_val'] >= 0.7])
                    uncertain = len(df[df['Type'].str.contains('Uncertain', na=False)])
                    
                    counts = df['Raw Type'].value_counts()
                    top_type = counts.index[0] if not counts.empty else "N/A"
                    
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Total Cells", total)
                    mc2.metric("High Confidence", f"{high_conf}/{total}")
                    mc3.metric("Most Frequent", top_type)
                    
                    # Distribution chart
                    fig = px.pie(
                        values=counts.values, names=counts.index,
                        color=counts.index,
                        color_discrete_map=CLASS_COLORS,
                        hole=0.4
                    )
                    fig.update_layout(
                        height=280, margin=dict(t=10, b=10, l=10, r=10),
                        legend=dict(font=dict(size=10)),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Data table (display-friendly columns only)
                    display_df = df[['ID', 'Type', 'Confidence', 'Box']].copy()
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Run pipeline to see results")
        
        # --- CROP INSPECTOR TAB ---
        with tab_crops:
            st.markdown("### 🔍 Crop Inspector — Kiểm tra AI đang nhìn thấy gì")
            st.caption("Mỗi crop bên dưới là 1 tế bào mà AI đã cắt ra và phân loại. Kiểm tra xem crop có đúng 1 cell không.")
            
            if st.session_state.crops and st.session_state.crop_labels:
                n_cols = 6
                crops = st.session_state.crops
                labels = st.session_state.crop_labels
                
                for row_start in range(0, len(crops), n_cols):
                    cols = st.columns(n_cols)
                    for j, col in enumerate(cols):
                        idx = row_start + j
                        if idx >= len(crops):
                            break
                        with col:
                            st.image(crops[idx], use_container_width=True)
                            cls_name, conf = labels[idx]
                            color = CLASS_COLORS.get(cls_name, "#666")
                            st.markdown(
                                f"<div style='text-align:center;font-size:0.75rem;'>"
                                f"<b>#{idx+1}</b><br>"
                                f"<span style='color:{color}'>{cls_name}</span><br>"
                                f"{conf:.0%}</div>",
                                unsafe_allow_html=True
                            )
            else:
                st.info("Run pipeline to inspect cell crops")
        
        # ═══════════════════════════════════════════════════════
        # AI CLINICAL INSIGHTS (Full Slide only)
        # ═══════════════════════════════════════════════════════
        if st.session_state.cls_df is not None and not st.session_state.cls_df.empty:
            st.divider()
            st.markdown("### 🤖 AI Clinical Insights")
            
            df = st.session_state.cls_df
            counts = df['Raw Type'].value_counts()
            
            # Check for critical cells
            critical_types = ["Fragmented", "Teardrops"]
            critical_found = df[df['Raw Type'].isin(critical_types)]
            
            if not critical_found.empty:
                st.markdown(
                    f'<div class="insight-card insight-danger">'
                    f'🚨 <b>Cảnh báo khẩn:</b> Phát hiện {len(critical_found)} tế bào bất thường '
                    f'({", ".join(critical_found["Raw Type"].unique())}). Cần tham vấn bác sĩ.</div>',
                    unsafe_allow_html=True
                )
            
            # Show insights for top types
            for cell_type in counts.index[:3]:
                info = CLINICAL_ADVICE.get(cell_type, CLINICAL_ADVICE["Rounded RBCs"])
                severity = info['severity']
                css_class = ["insight-ok", "insight-watch", "insight-warn", "insight-warn", "insight-danger"][min(severity, 4)]
                
                st.markdown(
                    f'<div class="insight-card {css_class}">'
                    f'{info["icon"]} <b>{cell_type}</b> — {counts[cell_type]} cells — '
                    f'<i>{info["status"]}</i><br>'
                    f'{info["msg"]} {info["advice"]}</div>',
                    unsafe_allow_html=True
                )
    
    else:
        # ═══════════════════════════════════════════════════════
        # SINGLE CELL LAYOUT
        # ═══════════════════════════════════════════════════════
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown("### 🖼️ Uploaded Cell")
            st.image(st.session_state.input_image, use_container_width=True)
        
        with col2:
            st.markdown("### 🏷️ Classification Result")
            
            if st.session_state.cls_df is not None:
                df = st.session_state.cls_df
                top = df.iloc[0]
                pred_type = top['Type']
                conf = top['Conf_val']
                
                # Big prediction display
                info = CLINICAL_ADVICE.get(pred_type, CLINICAL_ADVICE["Rounded RBCs"])
                color = CLASS_COLORS.get(pred_type, "#666")
                
                st.markdown(f"""
                <div style="text-align:center; padding: 20px; background: linear-gradient(135deg, #1a1a2e, #16213e);
                            border-radius: 16px; border: 2px solid {color}; margin-bottom: 16px;">
                    <div style="font-size: 3rem;">{info['icon']}</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {color}; margin: 8px 0;">{pred_type}</div>
                    <div style="font-size: 1rem; color: #aaa;">Confidence: {conf:.1%}</div>
                    <div style="font-size: 0.85rem; color: #888; margin-top: 4px;">{info['status']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Top-3 bar
                if len(df) > 1:
                    st.caption("Top-3 Predictions:")
                    for _, row in df.iterrows():
                        c = CLASS_COLORS.get(row['Type'], '#666')
                        pct = row['Conf_val']
                        st.markdown(
                            f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0;'>"
                            f"<span style='min-width:140px;font-size:0.85rem;'>{row['Type']}</span>"
                            f"<div style='flex:1;background:#333;border-radius:4px;height:20px;'>"
                            f"<div style='width:{pct*100:.0f}%;background:{c};height:100%;border-radius:4px;'></div>"
                            f"</div>"
                            f"<span style='font-size:0.85rem;min-width:50px;text-align:right;'>{pct:.1%}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                
                # Clinical insight
                st.divider()
                severity = info['severity']
                css_class = ["insight-ok", "insight-watch", "insight-warn", "insight-warn", "insight-danger"][min(severity, 4)]
                st.markdown(
                    f'<div class="insight-card {css_class}">'
                    f'<b>{info["msg"]}</b><br>{info["advice"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info("👈 Upload an image and click Classify to see results")

else:
    # Landing page
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px; color: #888;">
        <div style="font-size: 4rem; margin-bottom: 16px;">🔬</div>
        <h2 style="color: #ccc;">Ready to Analyze</h2>
        <p>Upload an image in the sidebar to begin RBC morphology analysis.</p>
        <div style="display: flex; justify-content: center; gap: 40px; margin-top: 32px;">
            <div style="text-align: center;">
                <div style="font-size: 2rem;">🖼️</div>
                <b>Full Slide</b><br>
                <span style="font-size: 0.85rem;">Upload tiêu bản toàn cảnh<br>→ Tách & phân loại từng cell</span>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem;">🧬</div>
                <b>Single Cell</b><br>
                <span style="font-size: 0.85rem;">Upload 1 cell đã crop<br>→ Phân loại trực tiếp</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
