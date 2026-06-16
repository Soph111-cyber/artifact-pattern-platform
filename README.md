# 🏺 Artifact Intelligence Lab

A Streamlit research prototype for global artifact pattern recognition, fragment completion, and cultural attribution.

## What it does

Users can upload an artifact pattern or fragment image. The platform extracts visual features and compares them with a global Excel-based motif database.

Main modules:

1. **Pattern feature recognition**
   - edge density
   - symmetry score
   - texture roughness
   - contour complexity
   - visual keyword tags

2. **Global cultural attribution**
   - uses `data/global_artifact_pattern_feature_database.xlsx`
   - returns top candidate cultures/styles, eras, regions, motif features, possible confusions, and source URLs

3. **Fragment restoration / 3D reconstruction plan**
   - suggests segmentation, feature matching, ICP alignment, SfM/MVS, COLMAP/Meshroom workflow

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this folder to GitHub.
2. Go to Streamlit Cloud.
3. Choose the GitHub repository.
4. Main file path: `app.py`.
5. Deploy.

## Important note

This is a research-demo platform, not a professional archaeological authentication system. Real attribution requires material, object type, measurements, provenance, excavation context, inscriptions, and expert-reviewed databases.
