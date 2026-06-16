
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import json
from pathlib import Path
import re

st.set_page_config(
    page_title="Artifact Intelligence Lab",
    page_icon="🏺",
    layout="wide",
)

# ✅ Fixed version:
# The app will search for the Excel database in BOTH places:
# 1. root folder: global_artifact_pattern_feature_database.xlsx
# 2. data folder: data/global_artifact_pattern_feature_database.xlsx
POSSIBLE_DB_PATHS = [
    Path("global_artifact_pattern_feature_database.xlsx"),
    Path("data/global_artifact_pattern_feature_database.xlsx"),
]

TEXT = {
    "中文": {
        "title": "🏺 Artifact Intelligence Lab｜全球文物纹样识别与修复平台",
        "subtitle": "上传纹样或文物碎片图片，平台将结合图像特征与全球纹样数据库，输出可能的文化来源、时代地域、纹样元素与修复/3D重建思路。",
        "upload": "上传纹样 / 文物碎片图片",
        "run": "开始分析",
        "db_status": "全球纹样数据库状态",
        "no_db": "未找到 Excel 数据库。请把 global_artifact_pattern_feature_database.xlsx 放在仓库根目录，和 app.py 同一级。",
        "original": "原始图片",
        "enhanced": "增强图 + 边缘提示",
        "visual_features": "图像特征提取",
        "pattern_elements": "识别出的纹样元素",
        "global_candidates": "全球文化来源候选",
        "reconstruction": "破碎文物修复与3D重建建议",
        "report": "综合报告",
        "download": "下载 JSON 报告",
        "warning": "注意：这是研究型原型平台，不是专业鉴定结论。真正判断需要结合材质、器型、尺寸、出土语境、铭文、博物馆/考古报告来源等证据。",
        "metadata": "可选补充信息",
        "material": "材质",
        "object_type": "器型/物件类型",
        "region_hint": "已知或猜测地区",
        "era_hint": "已知或猜测时代",
        "caption": "图片描述/备注",
        "top_k": "候选数量",
    },
    "English": {
        "title": "🏺 Artifact Intelligence Lab｜Global Artifact Pattern Recognition & Reconstruction",
        "subtitle": "Upload an artifact pattern or fragment image. The platform combines visual features with a global motif database to infer possible cultural sources, era-region context, pattern elements, and reconstruction ideas.",
        "upload": "Upload pattern / artifact fragment image",
        "run": "Run analysis",
        "db_status": "Global motif database status",
        "no_db": "Excel database not found. Please place global_artifact_pattern_feature_database.xlsx in the root folder, at the same level as app.py.",
        "original": "Original image",
        "enhanced": "Enhanced image + edge overlay",
        "visual_features": "Extracted visual features",
        "pattern_elements": "Recognized pattern elements",
        "global_candidates": "Global cultural attribution candidates",
        "reconstruction": "Fragment restoration & 3D reconstruction suggestions",
        "report": "Integrated report",
        "download": "Download JSON report",
        "warning": "Note: this is a research prototype, not a professional authentication result. Real attribution requires material, object type, measurements, provenance, inscription, excavation context, and museum/archaeological sources.",
        "metadata": "Optional metadata",
        "material": "Material",
        "object_type": "Object type",
        "region_hint": "Known or guessed region",
        "era_hint": "Known or guessed era",
        "caption": "Image description / notes",
        "top_k": "Number of candidates",
    }
}

def find_db_path():
    for path in POSSIBLE_DB_PATHS:
        if path.exists():
            return path
    return None

def safe_text(x):
    if pd.isna(x):
        return ""
    return str(x)

@st.cache_data
def load_database():
    db_path = find_db_path()
    if db_path is None:
        return None, None
    main = pd.read_excel(db_path, sheet_name="Pattern_Feature_DB")
    return main, db_path

def pil_to_cv2(img):
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_to_pil(arr):
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def resize_keep_ratio(img, max_side=900):
    w, h = img.size
    scale = min(max_side / max(w, h), 1)
    return img.resize((int(w * scale), int(h * scale)))

def preprocess(img):
    img = resize_keep_ratio(img)
    bgr = pil_to_cv2(img)
    denoise = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(denoise, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(denoise, 0.78, edge_bgr, 0.22, 0)
    return cv2_to_pil(overlay), gray, edges

def extract_features(img, gray, edges):
    h, w = gray.shape
    edge_density = float((edges > 0).sum() / edges.size)
    small = cv2.resize(gray, (256, 256))
    v_sym = 1 - np.mean(np.abs(small.astype(float) - cv2.flip(small, 1).astype(float))) / 255
    h_sym = 1 - np.mean(np.abs(small.astype(float) - cv2.flip(small, 0).astype(float))) / 255
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    contour_area_ratio = sum(cv2.contourArea(c) for c in contours) / (w * h)

    rgb = np.array(img.convert("RGB"))
    mean_rgb = rgb.reshape(-1, 3).mean(axis=0)
    red_warmth = float(mean_rgb[0] - mean_rgb[2])
    brightness = float(mean_rgb.mean())

    return {
        "edge_density": round(edge_density, 4),
        "vertical_symmetry": round(float(v_sym), 3),
        "horizontal_symmetry": round(float(h_sym), 3),
        "texture_roughness": round(lap_var, 2),
        "contour_count": int(contour_count),
        "contour_area_ratio": round(float(contour_area_ratio), 4),
        "mean_rgb": [round(float(x), 2) for x in mean_rgb],
        "red_warmth": round(red_warmth, 2),
        "brightness": round(brightness, 2),
    }

def infer_visual_tags(features):
    tags = []
    if max(features["vertical_symmetry"], features["horizontal_symmetry"]) > 0.72:
        tags += ["symmetry", "symmetrical", "axis", "mirror", "对称"]
    if features["edge_density"] > 0.09:
        tags += ["dense", "linear", "incised", "geometric", "密集", "线刻", "几何"]
    if features["contour_count"] > 120:
        tags += ["repeated", "tessellation", "interlace", "repetition", "重复", "连续"]
    if features["texture_roughness"] > 500:
        tags += ["relief", "rough", "carved", "embossed", "浮雕", "雕刻"]
    if features["red_warmth"] > 18:
        tags += ["red", "terracotta", "pottery", "warm", "红", "陶"]
    if features["brightness"] < 95:
        tags += ["dark", "bronze", "black", "深色", "青铜"]
    return list(dict.fromkeys(tags))

def recognize_elements(features):
    elements = []
    if max(features["vertical_symmetry"], features["horizontal_symmetry"]) > 0.72:
        elements.append({"element": "symmetrical composition / 对称构图", "confidence": 0.78})
    if features["edge_density"] > 0.09:
        elements.append({"element": "dense linear or incised motif / 密集线性或线刻纹样", "confidence": 0.72})
    if features["contour_count"] > 120:
        elements.append({"element": "repeated geometric unit / 重复几何单元", "confidence": 0.68})
    if features["texture_roughness"] > 500:
        elements.append({"element": "relief-like or carved surface / 浮雕感或雕刻表面", "confidence": 0.64})
    if not elements:
        elements.append({"element": "low-to-medium complexity motif / 中低复杂度纹样", "confidence": 0.50})
    return elements

def tokenize(s):
    s = safe_text(s).lower()
    return set(re.findall(r"[a-zA-Z][a-zA-Z\-]+|[\u4e00-\u9fff]+", s))

def row_score(row, visual_tags, metadata_text):
    search_cols = [
        "Key_motifs_EN", "关键纹样_CN", "Geometry_tags", "Color_palette_hints",
        "Material_surface_hints", "Model_positive_keywords", "Model_visual_features",
        "Potential_confusions", "Suggested_model_label", "Culture_or_style", "Era_period",
        "Macro_region", "Region_or_origin"
    ]
    row_text = " ".join(safe_text(row.get(c, "")) for c in search_cols)
    row_tokens = tokenize(row_text)
    meta_tokens = tokenize(metadata_text)
    tag_tokens = set(t.lower() for t in visual_tags)

    tag_hits = len(row_tokens & tag_tokens)
    meta_hits = len(row_tokens & meta_tokens)

    base = 0.35
    score = base + 0.08 * tag_hits + 0.05 * meta_hits

    try:
        seed = float(row.get("Attribution_confidence_seed", 0))
        if 0 <= seed <= 1:
            score = 0.65 * score + 0.35 * seed
    except Exception:
        pass

    return min(round(score, 3), 0.95), tag_hits, meta_hits

def global_match(db, visual_tags, metadata, top_k=5):
    metadata_text = " ".join(metadata.values())
    rows = []
    for _, row in db.iterrows():
        score, tag_hits, meta_hits = row_score(row, visual_tags, metadata_text)
        rows.append({
            "Score": score,
            "Culture_or_style": safe_text(row.get("Culture_or_style", "")),
            "Era_period": safe_text(row.get("Era_period", "")),
            "Macro_region": safe_text(row.get("Macro_region", "")),
            "Region_or_origin": safe_text(row.get("Region_or_origin", "")),
            "Key_motifs_EN": safe_text(row.get("Key_motifs_EN", "")),
            "关键纹样_CN": safe_text(row.get("关键纹样_CN", "")),
            "Model_visual_features": safe_text(row.get("Model_visual_features", "")),
            "Potential_confusions": safe_text(row.get("Potential_confusions", "")),
            "Source_URL": safe_text(row.get("Source_URL", "")),
            "Tag_hits": tag_hits,
            "Metadata_hits": meta_hits
        })
    out = pd.DataFrame(rows).sort_values("Score", ascending=False).head(top_k)
    return out.reset_index(drop=True)

def reconstruction_advice(features):
    advice = []
    if features["edge_density"] > 0.08:
        advice.append("边缘信息较丰富：可优先做断裂边界提取、边缘曲率匹配、纹样连续性拼合。")
        advice.append("Technical path: Canny/SAM segmentation → SIFT/ORB/LoFTR feature matching → ICP alignment if 3D data exists.")
    else:
        advice.append("边缘信息偏弱：建议补充侧面、背面、斜光照片，避免只凭单图判断碎片拼合。")
    if max(features["vertical_symmetry"], features["horizontal_symmetry"]) > 0.72:
        advice.append("纹样有较强对称性：可尝试镜像补全，但必须标记为假设区域。")
    else:
        advice.append("对称性不强：更适合用相似器物数据库做 reference-based completion。")
    advice.append("若要真正 3D 重建：拍摄 30–80 张多角度照片，使用 COLMAP/Meshroom 生成点云和 mesh，再做纹理贴图。")
    return advice

# Sidebar
st.sidebar.title("Settings")
lang = st.sidebar.radio("Language / 语言", ["中文", "English"], index=0)
T = TEXT[lang]
top_k = st.sidebar.slider(T["top_k"], 3, 10, 5)

db, db_path = load_database()

st.title(T["title"])
st.caption(T["subtitle"])
st.warning(T["warning"])

with st.expander(T["db_status"], expanded=True):
    if db is None:
        st.error(T["no_db"])
        st.write("Current app searches these locations:")
        for path in POSSIBLE_DB_PATHS:
            st.code(str(path))
    else:
        st.success(f"Loaded database: {len(db)} rows from {db_path}")
        st.dataframe(db.head(8), use_container_width=True)

uploaded = st.file_uploader(T["upload"], type=["jpg", "jpeg", "png"])

with st.expander(T["metadata"], expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        material = st.text_input(T["material"], placeholder="ceramic / bronze / textile / jade...")
    with c2:
        object_type = st.text_input(T["object_type"], placeholder="vessel / tile / textile / ornament...")
    with c3:
        region_hint = st.text_input(T["region_hint"], placeholder="Andes / Mediterranean / China...")
    with c4:
        era_hint = st.text_input(T["era_hint"], placeholder="Neolithic / Roman / Ming...")
    caption = st.text_area(T["caption"], placeholder="Any visible motifs, museum label, excavation context, etc.")

if uploaded and db is not None:
    img = Image.open(uploaded).convert("RGB")
    enhanced, gray, edges = preprocess(img)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(T["original"])
        st.image(img, use_container_width=True)
    with col2:
        st.subheader(T["enhanced"])
        st.image(enhanced, use_container_width=True)

    if st.button(T["run"], type="primary"):
        features = extract_features(img, gray, edges)
        visual_tags = infer_visual_tags(features)
        elements = recognize_elements(features)
        metadata = {
            "material": material,
            "object_type": object_type,
            "region_hint": region_hint,
            "era_hint": era_hint,
            "caption": caption,
        }
        candidates = global_match(db, visual_tags, metadata, top_k=top_k)
        advice = reconstruction_advice(features)

        st.header(T["visual_features"])
        a, b, c, d = st.columns(4)
        a.metric("Edge density", features["edge_density"])
        b.metric("Symmetry max", max(features["vertical_symmetry"], features["horizontal_symmetry"]))
        c.metric("Texture roughness", features["texture_roughness"])
        d.metric("Contour count", features["contour_count"])
        st.write("**Visual tags:**", ", ".join(visual_tags))

        st.header(T["pattern_elements"])
        st.dataframe(pd.DataFrame(elements), use_container_width=True)

        st.header(T["global_candidates"])
        st.dataframe(candidates, use_container_width=True)

        st.header(T["reconstruction"])
        for item in advice:
            st.write("- " + item)

        report = {
            "image_size": img.size,
            "visual_features": features,
            "visual_tags": visual_tags,
            "recognized_elements": elements,
            "metadata": metadata,
            "global_candidates": candidates.to_dict(orient="records"),
            "reconstruction_advice": advice,
            "important_note": "Prototype output only. Use with archaeological context and expert review."
        }

        st.header(T["report"])
        st.json(report)
        st.download_button(
            T["download"],
            data=json.dumps(report, ensure_ascii=False, indent=2),
            file_name="artifact_intelligence_report.json",
            mime="application/json"
        )
