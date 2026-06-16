# 🏺 Artifact Pattern Intelligence & Reconstruction Platform  
# 文物纹样智能分析与破碎文物修复平台

这是一个可部署到 GitHub + Streamlit Cloud 的研究型 Demo 平台，用于展示：

1. **破碎文物修复完整**：利用边缘、对称性、纹样连续性提出 3D 重建和碎片拼合方案  
2. **识别纹样特征和元素**：利用图像预处理、多模态分析思路提取纹样特征  
3. **判断时代、地域和文化来源**：基于纹样、颜色、纹理、对称性等信息生成候选判断和置信度  

> Important: 当前版本是 Demo/Prototype，不是专业考古鉴定工具。真正判断需要结合出土层位、材质、器型、工艺、铭文、数据库和专家知识。

---

## 🚀 How to Run Locally / 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Deploy on GitHub + Streamlit Cloud

### 1. Create a GitHub repository

```bash
git init
git add .
git commit -m "init artifact intelligence platform"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/artifact-pattern-platform.git
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Open Streamlit Cloud
2. Choose **New app**
3. Connect your GitHub repository
4. Main file path: `app.py`
5. Click **Deploy**

---

## 🧠 Model Upgrade Plan / 后续模型升级方向

### A. 破碎文物修复完整 / 3D Reconstruction

Current demo:
- Edge detection
- Symmetry-based completion hypothesis
- Reconstruction pipeline suggestion

Future:
- Use **SAM** to segment fragments
- Use **SIFT/ORB/LoFTR** for feature matching
- Use **ICP** for 3D alignment
- Use **COLMAP / Meshroom / NeRF / Gaussian Splatting** for 3D reconstruction
- Use archaeological reference objects for shape completion

### B. 纹样特征识别 / Multimodal Pattern Analysis

Current demo:
- Dominant colors
- Edge density
- Symmetry score
- Texture roughness
- Contour complexity

Future:
- Use **CLIP** for image-text embedding
- Use **BLIP** for image captioning
- Use **ViT/CNN** for motif classification
- Use **knowledge graph** linking motif → material → region → era

### C. 时代地域文化来源判断 / Cultural Attribution

Current demo:
- Heuristic candidate generation

Future:
- Fine-tune classifier with labeled artifact images
- Add metadata: material, dimensions, provenance, excavation context
- Use Bayesian model to output posterior probability
- Add uncertainty explanation rather than a single deterministic label

---

## 📁 Project Structure

```text
artifact-pattern-platform/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── predict.py
└── data/
    └── sample_schema.json
```

---

## 🔬 Suggested Dataset Schema / 数据格式建议

```json
{
  "artifact_id": "A0001",
  "image_path": "images/A0001.jpg",
  "material": "ceramic",
  "object_type": "vessel fragment",
  "motifs": ["geometric", "spiral", "symmetry"],
  "era": "Neolithic",
  "region": "Yellow River basin",
  "culture": "Yangshao-related",
  "source": "museum / field record / database",
  "confidence": 0.82
}
```

---

## ✨ Research Framing / 项目叙事

This platform is not trying to replace archaeologists.  
It tries to make archaeological reasoning more transparent:

- What visual evidence do we see?
- Which features support a certain cultural attribution?
- Where is the uncertainty?
- What extra evidence would change the conclusion?

中文表述：

这个平台不是为了用 AI 替代考古学家，而是希望把考古判断过程变得更透明：  
我们看到了什么证据？这些证据为什么支持某种判断？哪里仍然不确定？还需要什么材料才能改变结论？
