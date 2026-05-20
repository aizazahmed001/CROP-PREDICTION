# 🌾 Smart Agriculture Decision Support System

> A production-grade, multi-model AI system for precision farming decisions — combining
> Decision Trees, KMeans Clustering, and Linear Regression in a unified Tkinter desktop application.

---

## 📌 Project Overview

This system assists agronomists and farmers in making data-driven decisions about:

| Goal | Model | Output |
|---|---|---|
| Which crop to plant? | Decision Tree Classifier | Crop recommendation (22 classes) |
| What soil type is this? | KMeans Clustering | Soil zone / segment label |
| What yield can I expect? | Linear Regression | Estimated yield (kg/ha) |

The input features are standard soil and climate measurements:
**Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Soil pH, Rainfall**.

---

## 🗂️ Repository Structure

```
smart_agri/
├── data/
│   └── dataset.csv              ← Crop Recommendation Dataset (2200 samples, 22 crops)
├── src/
│   ├── preprocessing.py         ← Data loading, cleaning, scaling, train/test split
│   ├── models.py                ← Model factory functions (DT, KMeans, LinReg)
│   ├── train.py                 ← Full training + evaluation + serialisation pipeline
│   ├── gui.py                   ← Tkinter desktop application
│   └── utils.py                 ← Metrics, plots, report savers
├── models/
│   ├── decision_tree.pkl
│   ├── knn.pkl                  ← KMeans model (soil segmentation)
│   ├── linear_regression.pkl
│   ├── label_encoder.pkl
│   ├── scaler.pkl
│   └── pca.pkl
├── results/
│   ├── accuracy_report.txt
│   ├── regression_metrics.txt
│   ├── clustering_metrics.txt
│   ├── feature_importance.png
│   ├── cluster_plot.png
│   └── residual_plot.png
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                   │
│   dataset.csv  →  preprocessing.py  →  StandardScaler          │
│                    (clean · encode · scale · split)             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
  ┌───────────────┐  ┌─────────────┐  ┌──────────────────┐
  │ Decision Tree │  │   KMeans    │  │ Linear Regression│
  │  Classifier   │  │ Clustering  │  │ Yield Predictor  │
  │  (crop type)  │  │ (soil zone) │  │ (kg/ha estimate) │
  └───────┬───────┘  └──────┬──────┘  └────────┬─────────┘
          │                 │                   │
          └─────────────────┼───────────────────┘
                            ▼
              ┌─────────────────────────┐
              │    Tkinter GUI (gui.py) │
              │   Input → Predict → UI  │
              └─────────────────────────┘
```

---

## ⚙️ Installation

**Prerequisites:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/your-username/smart-agri-dss.git
cd smart-agri-dss

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1 – Train all models
```bash
python train.py
```
This trains Decision Tree, KMeans, and Linear Regression, saves all models to `/models`,
and saves all evaluation plots + metric reports to `/results`.

### Step 2 – Launch the GUI
```bash
python src/gui.py
```

Enter soil and climate values, press **PREDICT**, and instantly see:
- Recommended crop
- Soil zone classification
- Estimated yield in kg/ha

---

## 🧠 Model Descriptions

### 1. Decision Tree Classifier — Crop Recommendation
- **Algorithm:** CART (Gini impurity criterion)
- **Hyper-parameters:** `max_depth=10`, `min_samples_leaf=4`
- **Input:** 7 soil/climate features (standardised)
- **Output:** One of 22 crop classes
- **Why Decision Tree?** Interpretable, visualisable, handles multi-class with no modification

### 2. KMeans Clustering — Soil Segmentation
- **Algorithm:** KMeans with k-means++ initialisation
- **Clusters:** 5 soil zones
- **Visualisation:** 2-D PCA projection
- **Why KMeans?** Efficient unsupervised partitioning; 5 zones map to broad agricultural soil categories (sandy, clay, loam, peat, chalk)

### 3. Linear Regression — Yield Prediction
- **Algorithm:** Ordinary Least Squares with intercept
- **Input:** Same 7 standardised features
- **Output:** Continuous yield estimate (kg/ha)
- **Why Linear Regression?** Baseline interpretable predictor; strong R² on this feature space

---

## 📊 Evaluation Metrics Summary

| Model | Metric | Value |
|---|---|---|
| Decision Tree | Accuracy | **95.91%** |
| Decision Tree | Weighted F1 | **95.94%** |
| KMeans | Silhouette Score | **0.293** |
| Linear Regression | RMSE | **218.14 kg/ha** |
| Linear Regression | MAE | **173.58 kg/ha** |
| Linear Regression | R² | **0.9659** |

---

## 🖼️ Results Visualisation

| Plot | Description |
|---|---|
| `feature_importance.png` | Horizontal bar chart ranking 7 input features by Decision Tree Gini importance |
| `cluster_plot.png` | 2-D PCA scatter coloured by KMeans soil zone (5 clusters) |
| `residual_plot.png` | Predicted vs. residual scatter + residual histogram for the regression model |

---

## 🌍 Industrial Application — Smart Farming

This system maps directly to **precision agriculture** workflows:

1. **Soil testing sensors** (IoT) transmit real-time NPK/pH/moisture data
2. The **Decision Tree** recommends the optimal crop for current conditions
3. **KMeans** clusters the farm into soil management zones for targeted fertiliser application
4. **Linear Regression** forecasts expected yield, supporting supply-chain and insurance planning

Deployable on a Raspberry Pi + touchscreen kiosk at field stations with minimal compute.

---

## 🔭 Future Work

### 1. IoT Sensor Integration
Connect the prediction pipeline to live MQTT sensor feeds (soil probes, weather stations).
Real-time inference enables dynamic crop scheduling across a season.

### 2. Satellite Imagery + Deep Learning
Augment tabular features with spectral indices (NDVI, NDWI) extracted from Sentinel-2 imagery.
A CNN-LSTM hybrid can capture temporal crop health trajectories beyond single-point predictions.

---

## 📚 Dataset

**Crop Recommendation Dataset**
- 2,200 samples | 22 balanced crop classes (100 per class)
- Features: N, P, K, Temperature, Humidity, pH, Rainfall
- Source: Agricultural domain benchmark dataset

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
