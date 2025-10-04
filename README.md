# Amazon Delivery Time Prediction Project

## Overview
This repository demonstrates an end-to-end pipeline to predict Amazon delivery times from order and contextual features. It includes exploratory data analysis (EDA), feature engineering, model training and evaluation, and a simple inference/deployment script.

## Contents
- EDA.ipynb — data loading, cleaning, exploratory plots, and feature engineering.
- model_building.ipynb — train/validate models, compare algorithms, tune hyperparameters, evaluate on holdout set, and save the best model and preprocessing artifacts.
- deployment.py — small inference utility that loads preprocessing artifacts and the trained model to produce delivery time predictions.
- amazon_delivery.csv — (place dataset here) raw data used by notebooks.

## Summary of EDA (EDA.ipynb)
- Data ingestion and initial checks (missing values, types, basic stats).
- Cleaning steps: handle missing values, convert datetime fields, compute derived durations.
- Feature engineering highlights:
  - Distance buckets and numeric distance feature.
  - Encoded categorical features (shipping method, product size, traffic/weather indicators).
  - Temporal features from order/shipment timestamps (weekday, hour).
- Visual findings: delivery-time distribution, correlations between distance/traffic/agent-rating and delivery time, outlier identification and handling.

## Summary of Model Building (model_building.ipynb)
- Train/validation split (time-aware or random split as appropriate).
- Models explored: baseline (mean predictor), linear regression, tree-based models (Random Forest / Gradient Boosting), optionally XGBoost/LightGBM.
- Preprocessing pipeline:
  - Impute/transform numeric features, scale where needed.
  - One-hot / ordinal encode categorical features.
  - Persist encoders and transformers using joblib or pickle.
- Evaluation metrics: RMSE, MAE, R². Use residual analysis and validation curves to select the best model.
- Artifacts saved: trained model file (e.g., model.pkl or model.joblib), preprocessing pipeline (e.g., preprocessor.joblib), and a small JSON/YAML with model metadata.

## Deployment / Inference (deployment.py)
- Loads preprocessing pipeline and trained model.
- Exposes a function or CLI to accept a single order (or batch) as a dict/CSV and returns predicted delivery time.
- Example behavior:
  - Read input (dict/CSV), apply same transforms as training, call model.predict, and format output (estimated delivery time ± expected error).
- Adapt the script to run behind a small Flask/FastAPI server or as a serverless function for production usage.

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # or
   pip install pandas numpy scikit-learn joblib matplotlib seaborn
   ```
2. Place dataset `amazon_delivery.csv` in repository root.
3. Run EDA:
   - Open and run `EDA.ipynb` to reproduce data cleaning and visualization steps.
4. Train models:
   - Open `model_building.ipynb`, run training cells, and save trained artifacts.
5. Inference example (python):
   ```python
   # minimal usage example for deployment.py
   from deployment import predict_from_dict
   sample = {
     "distance_km": 12.3,
     "product_size": "medium",
     "shipping_method": "standard",
     "traffic_level": "moderate",
     "order_hour": 15,
     # ...other required features...
   }
   eta = predict_from_dict(sample)
   print("Predicted delivery time (hours):", eta)
   ```

## File map and tips
- EDA.ipynb: follow the first section to reproduce cleaning and create `features.csv`.
- model_building.ipynb: ensure paths to saved artifacts match those referenced in `deployment.py`.
- deployment.py: confirm the preprocessor and model filenames; adapt input parsing to your deployment environment.

## Notes & Extensions
- Consider time-dependent validation (e.g., rolling-window) if temporal drift exists.
- Add uncertainty estimates (prediction intervals) or quantile regression for reliability.
- For production, wrap `deployment.py` in a lightweight API (FastAPI) and containerize with Docker.


## Click on the Link Below to Visit the app
- https://deliverytimeamazon.streamlit.app/

## License
Educational / internal use.
