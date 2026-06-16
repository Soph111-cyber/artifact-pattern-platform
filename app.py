
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from io import BytesIO
from pathlib import Path
import json
import re

st.set_page_config(
    page_title="Artifact Intelligence Lab",
    page_icon="🏺",
    layout="wide",
)

DEFAULT_DB_PATH = Path("global_pattern_feature_template.xlsx")

# ----------------------------
# Text
# ----------------------------
TEXT = {
    "中文": {
        "title": "🏺 Artifact Intelligence Lab｜纹样补全 + 全球文化来源分析",
        "subtitle": "上传破损纹样图片，平台会自动识别缺失区域并进行纹样补全；再结合你上传的 Excel 全球纹样数据库，输出更科学的文化来源候选与证据解释。",
        "warning": "注意：这是研究型原型平台，不是专业鉴定结论。真正判断需要结合材质、器型、尺寸、出土语境、铭文、博物馆/考古报告来源等证据。",
        "upload_image": "① 上传破损纹样 / 文物碎片图片",
        "upload_excel": "② 上传纹样特征 Excel 数据库（可选）",
        "db_help": "如果不上传，系统会使用仓库自带的 global_pattern_feature_template.xlsx。你可以下载模板后继续扩展。",
        "image_repair": "A. 破损纹样补全",
        "analysis": "B. 纹样文化来源分析",
        "settings": "补全设置",
        "mask_method": "缺失区域识别方式",
        "threshold": "白色/空白区域阈值",
        "min_area": "最小缺失区域面积",
        "run_repair": "运行纹样补全",
        "run_analysis": "运行文化来源分析",
        "original": "原始图片",
        "mask": "自动识别的缺失区域 Mask",
        "repaired": "补全结果",
        "enhanced": "边缘/结构提示图",
        "download_repaired": "下载补全图片",
        "metadata": "可选补充信息",
        "material": "材质",
        "object_type": "器型/物件类型",
        "known_region": "已知/猜测地区",
        "known_era": "已知/猜测时代",
        "visual_caption": "你观察到的纹样描述，或大模型/老师给出的图片描述",
        "top_k": "候选数量",
        "visual_features": "图像特征",
        "visual_tags": "系统提取的视觉标签",
        "candidates": "全球文化来源候选",
        "evidence": "证据解释",
        "report": "综合 JSON 报告",
        "download_report": "下载分析报告",
        "template": "下载 Excel 数据库模板",
        "db_loaded": "数据库已加载",
        "db_missing": "未找到默认数据库，也未上传 Excel。请上传一个 Excel 数据库。",
        "sheet_notice": "Excel 最好包含 Pattern_Feature_DB 这个 sheet；如果没有，系统会读取第一个 sheet。",
        "how_it_works": "平台如何判断？",
    },
    "English": {
        "title": "🏺 Artifact Intelligence Lab｜Pattern Completion + Global Cultural Attribution",
        "subtitle": "Upload a damaged pattern image. The platform detects missing regions, completes the visual pattern, and compares extracted features with an uploaded Excel motif database to generate evidence-based cultural attribution candidates.",
        "warning": "Note: this is a research prototype, not a professional authentication result. Real attribution requires material, object type, measurements, provenance, inscription, excavation context, and museum/archaeological sources.",
        "upload_image": "① Upload damaged pattern / artifact fragment image",
        "upload_excel": "② Upload pattern-feature Excel database (optional)",
        "db_help": "If no Excel is uploaded, the app uses the bundled global_pattern_feature_template.xlsx. You can download and expand the template.",
        "image_repair": "A. Damaged Pattern Completion",
        "analysis": "B. Cultural Attribution Analysis",
        "settings": "Completion settings",
        "mask_method": "Missing-area detection mode",
        "threshold": "White/blank threshold",
        "min_area": "Minimum missing-region area",
        "run_repair": "Run pattern completion",
        "run_analysis": "Run attribution analysis",
        "original": "Original image",
        "mask": "Detected missing-region mask",
        "repaired": "Completed result",
        "enhanced": "Edge/structure guide",
        "download_repaired": "Download completed image",
        "metadata": "Optional metadata",
        "material": "Material",
        "object_type": "Object type",
        "known_region": "Known/guessed region",
        "known_era": "Known/guessed era",
        "visual_caption": "Your motif description, or a vision model/teacher's image description",
        "top_k": "Number of candidates",
        "visual_features": "Visual features",
        "visual_tags": "Extracted visual tags",
        "candidates": "Global cultural attribution candidates",
        "evidence": "Evidence explanation",
        "report": "Integrated JSON report",
        "download_report": "Download analysis report",
        "template": "Download Excel database template",
        "db_loaded": "Database loaded",
        "db_missing": "No default database found and no Excel uploaded. Please upload an Excel database.",
        "sheet_notice": "The Excel file should preferably contain a sheet named Pattern_Feature_DB. If not, the app reads the first sheet.",
        "how_it_works": "How does the platform reason?",
    }
}

# ----------------------------
# Utilities
# ----------------------------
def pil_to_rgb_array(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def rgb_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def image_to_png_bytes(img: Image.Image) -> bytes:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()

def resize_keep_ratio(img: Image.Image, max_side=1100) -> Image.Image:
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    return img.resize((int(w * scale), int(h * scale)))

def safe_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)

def tokenize(text: str):
    text = safe_text(text).lower()
    return set(re.findall(r"[a-zA-Z][a-zA-Z\-]+|[\u4e00-\u9fff]+", text))

# ----------------------------
# Excel database loading
# ----------------------------
@st.cache_data
def read_excel_database(uploaded_excel_bytes):
    if uploaded_excel_bytes is not None:
        excel = pd.ExcelFile(BytesIO(uploaded_excel_bytes))
    elif DEFAULT_DB_PATH.exists():
        excel = pd.ExcelFile(DEFAULT_DB_PATH)
    else:
        return None, None

    sheet_name = "Pattern_Feature_DB" if "Pattern_Feature_DB" in excel.sheet_names else excel.sheet_names[0]
    df = pd.read_excel(excel, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]
    return df, sheet_name

def pick_col(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

# ----------------------------
# Missing region detection and completion
# ----------------------------
def detect_missing_mask(img: Image.Image, threshold=238, min_area_ratio=0.004, mode="自动识别白色/空白缺角"):
    arr = pil_to_rgb_array(img)
    h, w = arr.shape[:2]

    # Basic blank/white detection: high RGB values and low-to-medium saturation.
    gray_like_blank = np.all(arr > threshold, axis=2)

    # Also catch near-gray background holes after preprocessing.
    diff = arr.max(axis=2) - arr.min(axis=2)
    bright_gray = (arr.mean(axis=2) > threshold - 12) & (diff < 30)
    raw = (gray_like_blank | bright_gray).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep large connected components, especially border-touching ones.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    mask = np.zeros((h, w), dtype=np.uint8)
    min_area = max(50, int(h * w * min_area_ratio))

    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        touches_border = x == 0 or y == 0 or (x + ww) >= w - 1 or (y + hh) >= h - 1
        if area >= min_area and touches_border:
            mask[labels == i] = 255

    # If no large border component is found, keep large blank components anyway.
    if mask.sum() == 0:
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                mask[labels == i] = 255

    # Slightly expand mask so completion covers boundary artifacts.
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
    return mask

def create_edge_overlay(img: Image.Image):
    arr = pil_to_rgb_array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    denoise = cv2.fastNlMeansDenoisingColored(bgr, None, 8, 8, 7, 21)
    gray = cv2.cvtColor(denoise, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    overlay = cv2.addWeighted(denoise, 0.78, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.22, 0)
    return rgb_to_pil(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)), gray, edges

def symmetry_guided_completion(img: Image.Image, mask: np.ndarray, use_radial=True, use_horizontal=True, use_vertical=True):
    """
    Pattern-aware completion:
    1. Treat mask as missing area.
    2. Copy plausible pixels from symmetric counterparts: 180-degree rotation, horizontal flip, vertical flip.
    3. Use OpenCV inpainting to blend remaining holes/boundaries.
    This is not deep generative restoration, but it is much better for mandala/repeated motifs than simple gray mask display.
    """
    arr = pil_to_rgb_array(img)
    result = arr.copy()
    missing = mask > 0

    # Replace white missing zones with neutral average first to avoid harsh artifacts.
    valid = ~missing
    if valid.sum() > 0:
        avg = np.median(arr[valid], axis=0)
        result[missing] = avg

    candidates = []
    if use_radial:
        candidates.append(("180° rotational symmetry", np.flip(np.flip(arr, axis=0), axis=1), np.flip(np.flip(mask, axis=0), axis=1)))
    if use_horizontal:
        candidates.append(("horizontal mirror symmetry", np.flip(arr, axis=1), np.flip(mask, axis=1)))
    if use_vertical:
        candidates.append(("vertical mirror symmetry", np.flip(arr, axis=0), np.flip(mask, axis=0)))

    still_missing = missing.copy()
    used = []
    for name, cand_img, cand_mask in candidates:
        # A pixel can be copied if the counterpart is not missing.
        source_valid = cand_mask == 0
        can_fill = still_missing & source_valid
        if can_fill.sum() > 0:
            result[can_fill] = cand_img[can_fill]
            still_missing[can_fill] = False
            used.append(name)

    # Inpaint unresolved pixels and boundary to make it smoother.
    remaining_mask = (still_missing.astype(np.uint8) * 255)
    if remaining_mask.sum() > 0:
        bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        bgr = cv2.inpaint(bgr, remaining_mask, 5, cv2.INPAINT_TELEA)
        result = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Blend boundary only.
    boundary = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
    boundary = cv2.subtract(boundary, cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=1))
    if boundary.sum() > 0:
        bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        soft = cv2.GaussianBlur(bgr, (5, 5), 0)
        boundary_bool = boundary > 0
        bgr[boundary_bool] = (0.65 * bgr[boundary_bool] + 0.35 * soft[boundary_bool]).astype(np.uint8)
        result = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb_to_pil(result), used

# ----------------------------
# Visual feature extraction
# ----------------------------
def extract_visual_features(img: Image.Image, mask=None):
    img = resize_keep_ratio(img)
    arr = pil_to_rgb_array(img)
    h, w = arr.shape[:2]

    # Ignore missing mask if provided for feature extraction.
    valid_mask = np.ones((h, w), dtype=bool)
    if mask is not None:
        resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        valid_mask = resized_mask == 0
        if valid_mask.sum() < h * w * 0.3:
            valid_mask = np.ones((h, w), dtype=bool)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 180)

    edge_density = float(((edges > 0) & valid_mask).sum() / max(valid_mask.sum(), 1))

    small = cv2.resize(gray, (256, 256))
    v_sym = 1 - np.mean(np.abs(small.astype(float) - cv2.flip(small, 1).astype(float))) / 255
    h_sym = 1 - np.mean(np.abs(small.astype(float) - cv2.flip(small, 0).astype(float))) / 255
    r_sym = 1 - np.mean(np.abs(small.astype(float) - cv2.rotate(small, cv2.ROTATE_180).astype(float))) / 255

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pixels = arr[valid_mask]
    if len(pixels) == 0:
        pixels = arr.reshape(-1, 3)
    mean_rgb = pixels.mean(axis=0)

    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    valid_hsv = hsv[valid_mask]
    saturation = float(valid_hsv[:, 1].mean()) if len(valid_hsv) else 0

    # Color hints
    color_tags = []
    r, g, b = mean_rgb
    if r > b + 20 and r > g - 5:
        color_tags += ["red", "warm", "terracotta", "saffron", "红", "暖色"]
    if b > r + 15:
        color_tags += ["blue", "indigo", "青蓝"]
    if r > 120 and b > 110 and g < 110:
        color_tags += ["purple", "violet", "紫色"]
    if r > 150 and g > 110 and b < 95:
        color_tags += ["gold", "yellow", "saffron", "金色", "黄色"]
    if saturation > 90:
        color_tags += ["saturated", "high contrast", "高饱和"]

    features = {
        "edge_density": round(edge_density, 4),
        "vertical_symmetry": round(float(v_sym), 3),
        "horizontal_symmetry": round(float(h_sym), 3),
        "rotational_symmetry_180": round(float(r_sym), 3),
        "texture_roughness": round(lap_var, 2),
        "contour_count": int(len(contours)),
        "mean_rgb": [round(float(x), 2) for x in mean_rgb],
        "mean_saturation": round(saturation, 2),
    }

    tags = []
    sym_max = max(v_sym, h_sym, r_sym)
    if sym_max > 0.62:
        tags += ["symmetry", "symmetrical", "mirror", "对称"]
    if r_sym > 0.58:
        tags += ["radial", "rotational", "mandala", "central medallion", "放射状", "曼荼罗", "中心圆章"]
    if len(contours) > 80 and edge_density > 0.04:
        tags += ["dense", "repeated", "ornamental", "floral", "petals", "重复", "花瓣", "装饰"]
    if edge_density > 0.08:
        tags += ["linear", "incised", "geometric", "线性", "几何"]
    if lap_var > 500:
        tags += ["carved", "relief", "rough", "浮雕", "雕刻"]
    tags += color_tags

    # Very useful for textile/printed-looking images.
    if saturation > 70 and len(contours) > 60:
        tags += ["textile", "print", "painted", "fabric", "织物", "印花"]

    tags = list(dict.fromkeys([t.lower() for t in tags]))
    return features, tags

# ----------------------------
# Cultural attribution scoring
# ----------------------------
def score_database(df, visual_tags, metadata_text, top_k=5):
    col_map = {
        "culture": pick_col(df, ["Culture_or_style", "culture", "style", "文化来源", "文化/风格"]),
        "era": pick_col(df, ["Era_period", "era", "period", "时代"]),
        "macro": pick_col(df, ["Macro_region", "macro region", "continent", "大区"]),
        "region": pick_col(df, ["Region_or_origin", "region", "origin", "地域", "地区"]),
        "motifs": pick_col(df, ["Key_motifs_EN", "motifs", "key motifs", "纹样", "核心纹样"]),
        "motifs_cn": pick_col(df, ["关键纹样_CN", "关键纹样"]),
        "geometry": pick_col(df, ["Geometry_tags", "geometry", "几何标签"]),
        "colors": pick_col(df, ["Color_palette_hints", "colors", "color", "颜色"]),
        "surface": pick_col(df, ["Material_surface_hints", "material", "surface", "材质"]),
        "keywords": pick_col(df, ["Model_positive_keywords", "keywords", "positive keywords", "关键词"]),
        "visual": pick_col(df, ["Model_visual_features", "visual features", "视觉特征"]),
        "confusions": pick_col(df, ["Potential_confusions", "confusions", "易混淆项"]),
        "seed": pick_col(df, ["Attribution_confidence_seed", "confidence", "seed confidence", "置信度"]),
        "source": pick_col(df, ["Source_URL", "source", "url", "来源"]),
        "notes": pick_col(df, ["Notes", "notes", "备注"]),
    }

    search_cols = [c for c in col_map.values() if c is not None]
    metadata_tokens = tokenize(metadata_text)
    tag_tokens = set([t.lower() for t in visual_tags])

    scored = []
    for idx, row in df.iterrows():
        row_text = " ".join(safe_text(row.get(c, "")) for c in search_cols)
        row_tokens = tokenize(row_text)

        tag_hits = sorted(list(row_tokens & tag_tokens))
        metadata_hits = sorted(list(row_tokens & metadata_tokens))

        # Weighted evidence. Visual tags and metadata both matter.
        score = 0.28 + 0.065 * len(tag_hits) + 0.055 * len(metadata_hits)

        # Strong boost when explicit region/culture appears in metadata.
        culture_text = " ".join([
            safe_text(row.get(col_map["culture"], "")) if col_map["culture"] else "",
            safe_text(row.get(col_map["region"], "")) if col_map["region"] else "",
            safe_text(row.get(col_map["macro"], "")) if col_map["macro"] else "",
        ]).lower()
        for token in metadata_tokens:
            if len(token) >= 4 and token in culture_text:
                score += 0.12

        # Seed confidence from the database.
        if col_map["seed"]:
            try:
                seed = float(row.get(col_map["seed"], 0))
                if 0 <= seed <= 1:
                    score = 0.70 * score + 0.30 * seed
            except Exception:
                pass

        score = min(score, 0.96)

        scored.append({
            "Score": round(float(score), 3),
            "Culture_or_style": safe_text(row.get(col_map["culture"], "")) if col_map["culture"] else f"Row {idx + 2}",
            "Era_period": safe_text(row.get(col_map["era"], "")) if col_map["era"] else "",
            "Macro_region": safe_text(row.get(col_map["macro"], "")) if col_map["macro"] else "",
            "Region_or_origin": safe_text(row.get(col_map["region"], "")) if col_map["region"] else "",
            "Key_motifs": safe_text(row.get(col_map["motifs"], "")) if col_map["motifs"] else "",
            "关键纹样": safe_text(row.get(col_map["motifs_cn"], "")) if col_map["motifs_cn"] else "",
            "Model_visual_features": safe_text(row.get(col_map["visual"], "")) if col_map["visual"] else "",
            "Matched_visual_tags": ", ".join(tag_hits),
            "Matched_metadata_terms": ", ".join(metadata_hits),
            "Potential_confusions": safe_text(row.get(col_map["confusions"], "")) if col_map["confusions"] else "",
            "Source_URL": safe_text(row.get(col_map["source"], "")) if col_map["source"] else "",
            "Notes": safe_text(row.get(col_map["notes"], "")) if col_map["notes"] else "",
        })

    out = pd.DataFrame(scored).sort_values("Score", ascending=False).head(top_k).reset_index(drop=True)
    return out, col_map

# ----------------------------
# UI
# ----------------------------
st.sidebar.title("⚙️ Settings")
lang = st.sidebar.radio("Language / 语言", ["中文", "English"], index=0)
T = TEXT[lang]

top_k = st.sidebar.slider(T["top_k"], 3, 10, 5)

st.title(T["title"])
st.caption(T["subtitle"])
st.warning(T["warning"])

with st.expander(T["how_it_works"], expanded=False):
    st.markdown(
        """
        **核心逻辑 / Core logic**

        1. **缺失区域识别**：检测图片中接近白色/透明/空白的大面积边缘区域，生成 mask。  
        2. **纹样补全**：优先用 180° 旋转、水平镜像、垂直镜像，从已有图案中寻找可复制结构，再用 inpainting 平滑边界。  
        3. **视觉特征提取**：提取对称性、旋转对称、边缘密度、轮廓数量、纹理粗糙度、颜色倾向。  
        4. **Excel 证据匹配**：把系统视觉标签 + 用户补充信息，与 Excel 中的 motif / geometry / color / material / keywords 匹配。  
        5. **候选输出**：输出 top-k 文化来源候选、匹配证据、潜在混淆项和来源链接。
        """
    )

uploaded_excel = st.file_uploader(T["upload_excel"], type=["xlsx", "xls"])
st.caption(T["db_help"])
if DEFAULT_DB_PATH.exists():
    with open(DEFAULT_DB_PATH, "rb") as f:
        st.download_button(T["template"], data=f.read(), file_name="global_pattern_feature_template.xlsx")

excel_bytes = uploaded_excel.read() if uploaded_excel is not None else None
df, sheet_name = read_excel_database(excel_bytes)
if df is not None:
    st.success(f"{T['db_loaded']}: {len(df)} rows, sheet = {sheet_name}")
    with st.expander("Preview Excel database / 预览 Excel 数据库", expanded=False):
        st.dataframe(df.head(12), use_container_width=True)
else:
    st.error(T["db_missing"])

uploaded_image = st.file_uploader(T["upload_image"], type=["jpg", "jpeg", "png", "webp"])

if uploaded_image is None:
    st.stop()

img = Image.open(uploaded_image)
img = resize_keep_ratio(img.convert("RGB"))

st.header(T["image_repair"])

with st.sidebar.expander(T["settings"], expanded=True):
    mask_method = st.selectbox(T["mask_method"], ["自动识别白色/空白缺角"], index=0)
    threshold = st.slider(T["threshold"], 210, 252, 238)
    min_area_ratio = st.slider(T["min_area"], 0.001, 0.03, 0.004, step=0.001)
    use_radial = st.checkbox("Use 180° rotational symmetry / 使用180°旋转对称", value=True)
    use_horizontal = st.checkbox("Use horizontal mirror / 使用水平镜像", value=True)
    use_vertical = st.checkbox("Use vertical mirror / 使用垂直镜像", value=True)

mask = detect_missing_mask(img, threshold=threshold, min_area_ratio=min_area_ratio, mode=mask_method)
edge_overlay, gray, edges = create_edge_overlay(img)

if "repaired_img" not in st.session_state:
    st.session_state.repaired_img = None
if "repair_sources" not in st.session_state:
    st.session_state.repair_sources = []

c1, c2 = st.columns(2)
with c1:
    st.subheader(T["original"])
    st.image(img, use_container_width=True)
with c2:
    st.subheader(T["mask"])
    st.image(mask, use_container_width=True, clamp=True)

if st.button(T["run_repair"], type="primary"):
    repaired_img, used_sources = symmetry_guided_completion(
        img,
        mask,
        use_radial=use_radial,
        use_horizontal=use_horizontal,
        use_vertical=use_vertical,
    )
    st.session_state.repaired_img = repaired_img
    st.session_state.repair_sources = used_sources

if st.session_state.repaired_img is not None:
    c3, c4 = st.columns(2)
    with c3:
        st.subheader(T["repaired"])
        st.image(st.session_state.repaired_img, use_container_width=True)
    with c4:
        st.subheader(T["enhanced"])
        st.image(edge_overlay, use_container_width=True)
        st.write("Completion evidence / 补全依据：", ", ".join(st.session_state.repair_sources) if st.session_state.repair_sources else "inpainting only")
    st.download_button(
        T["download_repaired"],
        data=image_to_png_bytes(st.session_state.repaired_img),
        file_name="completed_pattern.png",
        mime="image/png",
    )

st.header(T["analysis"])

with st.expander(T["metadata"], expanded=True):
    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        material = st.text_input(T["material"], placeholder="textile / ceramic / bronze / paper...")
    with cc2:
        object_type = st.text_input(T["object_type"], placeholder="textile / vessel / tile / manuscript...")
    with cc3:
        known_region = st.text_input(T["known_region"], placeholder="India / Islamic world / Andes...")
    with cc4:
        known_era = st.text_input(T["known_era"], placeholder="ancient / medieval / modern...")
    visual_caption = st.text_area(
        T["visual_caption"],
        placeholder="例如：purple-gold radial mandala, lotus petals, concentric circular medallion, Indian textile pattern..."
    )

analysis_img = st.session_state.repaired_img if st.session_state.repaired_img is not None else img

if st.button(T["run_analysis"], type="primary"):
    if df is None:
        st.error(T["db_missing"])
        st.stop()

    features, visual_tags = extract_visual_features(analysis_img, mask=mask)
    metadata_text = " ".join([material, object_type, known_region, known_era, visual_caption])
    candidates, col_map = score_database(df, visual_tags, metadata_text, top_k=top_k)

    st.subheader(T["visual_features"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Edge density", features["edge_density"])
    m2.metric("Rotational symmetry", features["rotational_symmetry_180"])
    m3.metric("Texture roughness", features["texture_roughness"])
    m4.metric("Contour count", features["contour_count"])

    st.subheader(T["visual_tags"])
    st.write(", ".join(visual_tags))

    st.subheader(T["candidates"])
    st.dataframe(candidates, use_container_width=True)

    st.subheader(T["evidence"])
    if len(candidates) > 0:
        top = candidates.iloc[0].to_dict()
        st.markdown(f"### Top candidate: {top.get('Culture_or_style', '')}")
        st.write("**Score:**", top.get("Score", ""))
        st.write("**Era / Period:**", top.get("Era_period", ""))
        st.write("**Region:**", top.get("Region_or_origin", ""))
        st.write("**Matched visual tags:**", top.get("Matched_visual_tags", ""))
        st.write("**Matched metadata terms:**", top.get("Matched_metadata_terms", ""))
        st.write("**Potential confusions:**", top.get("Potential_confusions", ""))
        st.write("**Source:**", top.get("Source_URL", ""))

    report = {
        "visual_features": features,
        "visual_tags": visual_tags,
        "metadata": {
            "material": material,
            "object_type": object_type,
            "known_region": known_region,
            "known_era": known_era,
            "visual_caption": visual_caption,
        },
        "repair_sources": st.session_state.repair_sources,
        "global_candidates": candidates.to_dict(orient="records"),
        "database_sheet_used": sheet_name,
        "important_note": "Prototype output only. Use archaeological context and expert review."
    }

    st.subheader(T["report"])
    st.json(report)
    st.download_button(
        T["download_report"],
        data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name="artifact_analysis_report.json",
        mime="application/json",
    )
