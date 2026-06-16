# 🏺 Artifact Intelligence Lab  
## Pattern Completion + Global Cultural Attribution

This is a Streamlit research prototype for:

1. Completing damaged artifact/pattern images
2. Uploading a custom Excel database of global cultural motif features
3. Generating evidence-based cultural attribution candidates

## Main functions

### A. Damaged pattern completion
The app detects missing white/blank areas and uses:

- 180° rotational symmetry
- horizontal mirror symmetry
- vertical mirror symmetry
- OpenCV inpainting boundary blending

This is useful for mandala-like, textile-like, tiled, or repeated ornament patterns.

### B. Excel-based cultural attribution
You can upload your own `.xlsx` database. If no Excel is uploaded, the app uses:

```text
global_pattern_feature_template.xlsx
```

The recommended sheet name is:

```text
Pattern_Feature_DB
```

Important columns:

```text
Culture_or_style
Macro_region
Region_or_origin
Era_period
Key_motifs_EN
关键纹样_CN
Geometry_tags
Color_palette_hints
Material_surface_hints
Model_positive_keywords
Model_visual_features
Potential_confusions
Attribution_confidence_seed
Source_URL
Notes
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Upload these files to the root of your GitHub repository:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `global_pattern_feature_template.xlsx`
2. In Streamlit Cloud, set:
   - Main file path: `app.py`
3. Deploy.

## Important note

This platform is not a professional authentication system. It generates hypotheses and evidence trails. Real cultural attribution requires material, object type, measurements, provenance, excavation context, inscriptions, museum records, and expert review.
