# 🌍 Tourism Experience Analytics

## Project Overview
A complete Machine Learning pipeline that analyzes tourism data to provide
personalized recommendations, predict user satisfaction, and classify
visitor behavior.

---

## 🎯 Objectives
1. **Regression** → Predict attraction ratings (1–5)
2. **Classification** → Predict visit mode (Business/Couples/Family/Friends/Solo)
3. **Recommendation** → Suggest attractions using Content & Collaborative Filtering

---

## 📁 Project Structure
```
tourism_project/
├── data/                  → Raw xlsx files + master_dataset.csv + plots
├── models/                → Trained ML models (.pkl files)
├── 01_data_loading.ipynb  → Data cleaning & merging
├── 02_EDA.ipynb           → Exploratory Data Analysis
├── 03_model_building.ipynb→ ML model training
└── app.py                 → Streamlit web application
```

---

## 📊 Dataset
- 9 tables: Transaction, User, City, Country, Region,
  Continent, Item, Type, Mode
- 52,922 records after cleaning
- Source: Tourism Dataset (Google Drive)

---

## 🔍 Key EDA Findings
- 45% of users gave Rating 5 (highly positive)
- Couples is the most common visit mode (21,617)
- Australia is the top visiting country (13,000+ visits)
- Nature & Wildlife Areas is the most visited attraction type
- Peak tourism season: July–August

---

## 🤖 ML Models

| Model | Algorithm | Metric |
|-------|-----------|--------|
| Regression | Random Forest | R², RMSE |
| Classification | Random Forest | Accuracy, F1 |
| Recommendation | Cosine Similarity | Content + Collaborative |

---

## 🚀 How to Run

### Install dependencies
pip install pandas openpyxl numpy matplotlib seaborn scikit-learn streamlit

### Run Streamlit App
streamlit run app.py

---

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (ML models)
- Matplotlib, Seaborn (Visualization)
- Streamlit (Web App)
Note: Large model files (regression_model.pkl, 
classification_model.pkl) are not included due to 
GitHub file size limits. Run 03_model_building.ipynb 
to regenerate them.
---

## 📌 Business Use Cases
1. Personalized attraction recommendations
2. Tourism trend analytics
3. Customer segmentation by visit mode
4. Improving customer retention
