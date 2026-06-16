
import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import pandas as pd
import cv2
import io
import json
from pathlib import Path

# =========================================================
# Artifact Pattern Intelligence Platform
# 文物纹样智能分析与破碎文物修复平台
# ---------------------------------------------------------
# This Streamlit app is designed as a research-demo platform.
# Later, you can replace the heuristic modules with real models:
# - 3D reconstruction: COLMAP / NeRF / Gaussian Splatting / Meshroom
# - Multimodal analysis: CLIP / BLIP / SAM / custom CNN / Vision Transformer
# - Culture-era classifier: fine-tuned image classifier + metadata database
# =========================================================

st.set_page_config(
    page_title="Artifact Pattern Intelligence",
    page_icon="🏺",
    layout="wide",
)

# -----------------------------
# Bilingual UI dictionary
# -----------------------------
TEXT = {
    "中文": {
        "title": "🏺 文物纹样智能分析与破碎文物修复平台",
        "subtitle": "利用 3D 重建、多模态分析与纹样识别，辅助判断文物的完整形态、纹样特征、时代地域与文化来源。",
        "upload": "上传纹样 / 文物碎片图片",
        "upload_help": "支持 JPG、PNG、JPEG。若有多张碎片图，可逐张上传并记录分析结果。",
        "sidebar_title": "⚙️ 平台设置",
        "language": "界面语言",
        "confidence_mode": "置信度模式",
        "confidence_mode_help": "Demo 使用启发式规则生成结果；未来可替换为真实模型输出。",
        "analysis": "开始分析",
        "input_preview": "原始图片",
        "processed_preview": "增强后的图片",
        "module1": "① 破碎文物修复完整 / 3D重建思路",
        "module2": "② 纹样特征与元素识别 / 多模态分析",
        "module3": "③ 时代、地域与文化来源判断",
        "summary": "综合判断报告",
        "download": "下载分析报告 JSON",
        "no_file": "请先上传一张文物纹样或碎片图片。",
        "research_note": "研究提示",
        "warning": "这是演示平台，不应替代专业考古鉴定。真正判断需要结合出土层位、材质、器型、工艺、铭文和可靠数据库。",
        "github": "GitHub 部署说明",
    },
    "English": {
        "title": "🏺 Artifact Pattern Intelligence & Reconstruction Platform",
        "subtitle": "A research-demo platform using 3D reconstruction, multimodal pattern analysis, and cultural attribution to assist artifact interpretation.",
        "upload": "Upload pattern / artifact fragment image",
        "upload_help": "Supports JPG, PNG, JPEG. If you have multiple fragments, upload them one by one and save each report.",
        "sidebar_title": "⚙️ Platform Settings",
        "language": "Interface Language",
        "confidence_mode": "Confidence Mode",
        "confidence_mode_help": "This demo uses heuristic scoring. You can later replace it with real model predictions.",
        "analysis": "Run Analysis",
        "input_preview": "Original Image",
        "processed_preview": "Enhanced Image",
        "module1": "① Fragment Completion / 3D Reconstruction Plan",
        "module2": "② Pattern Feature & Element Recognition / Multimodal Analysis",
        "module3": "③ Era, Region & Cultural Attribution",
        "summary": "Integrated Interpretation Report",
        "download": "Download JSON Report",
        "no_file": "Please upload an artifact pattern or fragment image first.",
        "research_note": "Research Note",
        "warning": "This is a demo platform and should not replace professional archaeological authentication. Real attribution requires stratigraphy, material, form, craft, inscriptions, and verified databases.",
        "github": "GitHub Deployment Guide",
    }
}

# -----------------------------
# Utility functions
# -----------------------------
def pil_to_cv2(img: Image.Image):
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_to_pil(arr):
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(arr)

def resize_keep_ratio(img: Image.Image, max_side=900):
    w, h = img.size
    scale = min(max_side / max(w, h), 1)
    return img.resize((int(w * scale), int(h * scale)))

def enhance_image(img: Image.Image):
    """Basic preprocessing: grayscale, denoise, edge enhancement."""
    img = resize_keep_ratio(img)
    cv_img = pil_to_cv2(img)
    denoise = cv2.fastNlMeansDenoisingColored(cv_img, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(denoise, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(denoise, 0.75, edges_bgr, 0.25, 0)
    return cv2_to_pil(overlay), edges, gray

def extract_color_features(img: Image.Image):
    arr = np.array(img.convert("RGB"))
    pixels = arr.reshape(-1, 3)
    mean_rgb = pixels.mean(axis=0)
    std_rgb = pixels.std(axis=0)

    # Dominant colors by k-means
    pixels_small = pixels[np.random.choice(len(pixels), min(8000, len(pixels)), replace=False)]
    pixels_small = np.float32(pixels_small)
    K = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
    _, labels, centers = cv2.kmeans(pixels_small, K, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    counts = np.bincount(labels.flatten(), minlength=K)
    order = np.argsort(-counts)
    dominant = centers[order].tolist()
    proportions = (counts[order] / counts.sum()).round(3).tolist()

    return {
        "mean_rgb": mean_rgb.round(2).tolist(),
        "rgb_std": std_rgb.round(2).tolist(),
        "dominant_colors_rgb": dominant,
        "dominant_color_proportions": proportions
    }

def extract_shape_texture_features(img: Image.Image, edges, gray):
    h, w = gray.shape
    edge_density = float(np.sum(edges > 0) / edges.size)

    # Symmetry score: compare image with horizontal and vertical flipped versions
    gray_resized = cv2.resize(gray, (256, 256))
    v_flip = cv2.flip(gray_resized, 1)
    h_flip = cv2.flip(gray_resized, 0)
    vertical_sym = 1 - np.mean(np.abs(gray_resized.astype(float) - v_flip.astype(float))) / 255
    horizontal_sym = 1 - np.mean(np.abs(gray_resized.astype(float) - h_flip.astype(float))) / 255

    # Texture roughness via Laplacian variance
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Contour complexity
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    contour_area = sum(cv2.contourArea(c) for c in contours)
    contour_area_ratio = float(contour_area / (w * h))

    return {
        "edge_density": round(edge_density, 4),
        "vertical_symmetry_score": round(float(vertical_sym), 3),
        "horizontal_symmetry_score": round(float(horizontal_sym), 3),
        "texture_laplacian_variance": round(lap_var, 2),
        "contour_count": int(contour_count),
        "contour_area_ratio": round(contour_area_ratio, 4)
    }

def infer_pattern_elements(features):
    """Heuristic element recognition. Replace with real model later."""
    edge = features["edge_density"]
    v_sym = features["vertical_symmetry_score"]
    h_sym = features["horizontal_symmetry_score"]
    rough = features["texture_laplacian_variance"]
    contours = features["contour_count"]

    elements = []

    if v_sym > 0.72 or h_sym > 0.72:
        elements.append(("对称构图 / symmetrical composition", 0.78))
    if edge > 0.09:
        elements.append(("密集线刻 / dense incised lines", 0.72))
    if contours > 120:
        elements.append(("重复几何单元 / repeated geometric units", 0.68))
    if rough > 450:
        elements.append(("高纹理起伏 / rough or relief-like texture", 0.65))
    if edge < 0.045 and rough < 250:
        elements.append(("大面积色块 / broad color fields", 0.61))

    if not elements:
        elements.append(("低复杂度纹样 / low-complexity motif", 0.55))

    return elements

def infer_reconstruction_plan(features):
    """Generate a technical plan for fragment restoration and 3D reconstruction."""
    edge = features["edge_density"]
    contours = features["contour_count"]
    sym = max(features["vertical_symmetry_score"], features["horizontal_symmetry_score"])

    if edge > 0.08:
        fragment_match = "可优先利用边缘曲率、断裂线轮廓和纹样连续性进行碎片拼合。"
        tech = "建议使用边缘检测 + 特征点匹配 + ICP 配准；若有多角度照片，可进入 SfM/MVS 三维重建流程。"
    else:
        fragment_match = "边缘信息较弱，单图拼合不稳定，应补充侧面、背面和不同光照照片。"
        tech = "建议先做图像增强、材质分割，再结合人工标注断裂边界。"

    if sym > 0.72:
        completion = "纹样可能具有较强对称性，可用镜像补全作为初步复原假设。"
    else:
        completion = "对称性不强，补全时应避免直接镜像，需要依赖相似器物数据库。"

    return {
        "fragment_matching": fragment_match,
        "completion_hypothesis": completion,
        "recommended_pipeline": [
            "1. 图片预处理：去噪、矫正、增强边缘",
            "2. 断裂边界提取：Canny/SAM/人工标注",
            "3. 特征匹配：SIFT/ORB/深度特征匹配",
            "4. 三维重建：多角度图像 → SfM → 稠密点云 → mesh",
            "5. 完整性推测：利用对称性、纹样连续性和数据库相似器物",
            "6. 输出：复原草图、置信度、可疑区域标注"
        ],
        "technical_note": tech
    }

def infer_culture_era(features, color_features, elements):
    """
    Demo heuristic classifier.
    Later replace with trained model:
        prediction = model.predict(image, metadata)
    """
    edge = features["edge_density"]
    sym = max(features["vertical_symmetry_score"], features["horizontal_symmetry_score"])
    rough = features["texture_laplacian_variance"]
    mean_rgb = np.array(color_features["mean_rgb"])

    candidates = []

    # These categories are intentionally broad and cautious.
    # They are not professional authentication results.
    if edge > 0.09 and sym > 0.68:
        candidates.append({
            "label": "商周青铜器几何/兽面纹传统 | Shang-Zhou bronze geometric/taotie-related tradition",
            "confidence": 0.64,
            "reason": "边缘密度高、对称性较强，符合部分青铜器纹样常见的密集线条与轴对称特征。"
        })
    if rough > 500 and edge > 0.06:
        candidates.append({
            "label": "汉唐陶俑/瓦当/浮雕类装饰传统 | Han-Tang ceramic, tile-end, or relief decorative tradition",
            "confidence": 0.57,
            "reason": "纹理起伏较高，可能对应浮雕、模印或风化表面。"
        })
    if sym > 0.75 and edge < 0.08:
        candidates.append({
            "label": "新石器彩陶几何纹传统 | Neolithic painted pottery geometric tradition",
            "confidence": 0.56,
            "reason": "整体对称感较强、线条密度中等，可能接近几何化彩陶装饰。"
        })
    if mean_rgb[0] > mean_rgb[2] + 15 and edge < 0.08:
        candidates.append({
            "label": "陶器/彩陶暖色系装饰 | pottery or painted pottery warm-color tradition",
            "confidence": 0.52,
            "reason": "图像整体偏暖色，可能与陶质或红褐色颜料相关。"
        })
    if not candidates:
        candidates.append({
            "label": "待定：需要更多角度、材质和出土信息 | Undetermined: more metadata required",
            "confidence": 0.45,
            "reason": "仅凭单张图片无法稳定判断时代地域，需要结合材质、器型、尺寸、来源和考古语境。"
        })

    candidates = sorted(candidates, key=lambda x: x["confidence"], reverse=True)
    return candidates[:3]

def make_report(img, color_features, shape_features, elements, reconstruction, attribution):
    return {
        "platform": "Artifact Pattern Intelligence & Reconstruction Platform",
        "image_size": img.size,
        "color_features": color_features,
        "shape_texture_features": shape_features,
        "recognized_pattern_elements": [
            {"element": e, "confidence": c} for e, c in elements
        ],
        "fragment_completion_3d_reconstruction": reconstruction,
        "era_region_culture_candidates": attribution,
        "important_warning": "Demo result only. Professional attribution requires archaeological context, material analysis, typology, stratigraphy, inscriptions, and verified reference datasets."
    }

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Settings / 设置")
lang = st.sidebar.radio("Language / 语言", ["中文", "English"], index=0)
T = TEXT[lang]

st.sidebar.markdown("---")
confidence_mode = st.sidebar.selectbox(
    T["confidence_mode"],
    ["Demo heuristic model", "Future real model API"],
)
st.sidebar.caption(T["confidence_mode_help"])

st.sidebar.markdown("---")
st.sidebar.info(
    "Model interface reserved:\n"
    "`models/predict.py`\n\n"
    "You can later connect CLIP, BLIP, SAM, CNN, ViT, NeRF, COLMAP, or a custom archaeological database."
)

# -----------------------------
# Main page
# -----------------------------
st.title(T["title"])
st.caption(T["subtitle"])

st.warning(T["warning"])

uploaded = st.file_uploader(
    T["upload"],
    type=["jpg", "jpeg", "png"],
    help=T["upload_help"]
)

if uploaded is None:
    st.info(T["no_file"])
    st.markdown("### " + T["github"])
    st.code(
        "git init\n"
        "git add .\n"
        "git commit -m \"init artifact intelligence platform\"\n"
        "git branch -M main\n"
        "git remote add origin https://github.com/YOUR_USERNAME/artifact-pattern-platform.git\n"
        "git push -u origin main\n"
    )
    st.stop()

img = Image.open(uploaded).convert("RGB")
enhanced, edges, gray = enhance_image(img)

col1, col2 = st.columns(2)
with col1:
    st.subheader(T["input_preview"])
    st.image(img, use_container_width=True)
with col2:
    st.subheader(T["processed_preview"])
    st.image(enhanced, use_container_width=True)

if st.button(T["analysis"], type="primary"):
    color_features = extract_color_features(img)
    shape_features = extract_shape_texture_features(img, edges, gray)
    elements = infer_pattern_elements(shape_features)
    reconstruction = infer_reconstruction_plan(shape_features)
    attribution = infer_culture_era(shape_features, color_features, elements)
    report = make_report(img, color_features, shape_features, elements, reconstruction, attribution)

    st.markdown("---")

    # Module 1
    st.header(T["module1"])
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Fragment Matching / 碎片拼合判断**")
        st.write(reconstruction["fragment_matching"])
        st.markdown("**Completion Hypothesis / 补全假设**")
        st.write(reconstruction["completion_hypothesis"])
        st.markdown("**Technical Note / 技术说明**")
        st.write(reconstruction["technical_note"])
    with c2:
        st.markdown("**Recommended Pipeline / 推荐流程**")
        for step in reconstruction["recommended_pipeline"]:
            st.write(step)

    # Module 2
    st.header(T["module2"])
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Edge Density / 边缘密度", shape_features["edge_density"])
        st.metric("Contour Count / 轮廓数量", shape_features["contour_count"])
    with m2:
        st.metric("Vertical Symmetry / 垂直对称", shape_features["vertical_symmetry_score"])
        st.metric("Horizontal Symmetry / 水平对称", shape_features["horizontal_symmetry_score"])
    with m3:
        st.metric("Texture Roughness / 纹理起伏", shape_features["texture_laplacian_variance"])
        st.metric("Contour Area Ratio / 轮廓面积比", shape_features["contour_area_ratio"])

    st.markdown("**Recognized Pattern Elements / 识别出的纹样元素**")
    element_df = pd.DataFrame(elements, columns=["Element / 元素", "Confidence / 置信度"])
    st.dataframe(element_df, use_container_width=True)

    st.markdown("**Dominant Colors / 主色提取**")
    color_cols = st.columns(len(color_features["dominant_colors_rgb"]))
    for i, (rgb, prop) in enumerate(zip(color_features["dominant_colors_rgb"], color_features["dominant_color_proportions"])):
        with color_cols[i]:
            swatch = Image.new("RGB", (100, 60), tuple(rgb))
            st.image(swatch, caption=f"RGB {rgb}\n{prop}", use_container_width=True)

    # Module 3
    st.header(T["module3"])
    for cand in attribution:
        st.markdown(f"### {cand['label']}")
        st.progress(float(cand["confidence"]))
        st.write("**Reason / 理由：** " + cand["reason"])

    # Summary
    st.header(T["summary"])
    st.json(report)

    st.download_button(
        T["download"],
        data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name="artifact_pattern_analysis_report.json",
        mime="application/json"
    )

    st.info(
        "下一步升级方向：接入真实数据集；建立“纹样—器型—材质—时代—地域—文化来源”的知识库；"
        "用 CLIP/ViT 做图像嵌入检索，用 Bayesian model 表达不确定性。"
    )
