# 🔐 Credit Card Fraud Detection System  
### Machine Learning + Streamlit | Ensemble Model | Kaggle Dataset

This project is a **production-ready Credit Card Fraud Detection System** built using  
**Logistic Regression, Random Forest, Gradient Boosting**, combined into a weighted **ensemble model**.  
It includes a full **Streamlit UI**, **batch prediction**, **explainable outputs**, **model evaluation**,  
and **Kaggle credit card fraud dataset integration**.

---

## 🚀 Features

### 🧠 **Machine Learning**
- Ensemble of **Logistic Regression (30%)**, **Random Forest (35%)**, **Gradient Boosting (35%)**
- Real Kaggle dataset (284,807 transactions)
- SMOTE + Random Under-Sampling for handling class imbalance
- Feature engineering (Amount, Time, Geo-location, Age, Distance, etc.)
- Automatic model retraining & saving (`trained_model.pkl`)

### 🌐 **Streamlit Frontend**
- Single transaction fraud prediction  
- Batch prediction via CSV upload  
- Model performance dashboards (AUC, F1, Confusion Matrix, Feature Importance)  
- Dataset info & analytics  
- Full UI styling with custom CSS

---

## 📂 **Project Structure**
├── app.py        # Streamlit frontend GUI                          
├── backend/                                               
│ ├── ml_pipeline.py     # Full ML pipeline, training, evaluation            
│ └── trained_model.pkl   # Saved model (auto-generated)            
├── requirements.txt                     
└── QUICK_START.md              

---

## ⚙️ **Installation & Quick Start**

### Create environment & install dependencies  

```bash
python -m venv venv
venv\Scripts\activate       # Windows
# OR
source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

streamlit run app.py

```
---
## 🧪 Technologies Used

- **Python** – Core programming language for ML and backend logic  
- **Streamlit** – Interactive UI for real-time fraud detection  
- **Scikit-learn** – Training and evaluating machine learning models  
- **Imbalanced-Learn** – Handling class imbalance using SMOTE & undersampling  
- **NumPy / Pandas** – Data manipulation, preprocessing, and feature engineering  
- **Plotly** – Interactive visualizations and performance graphs  

---

## 📊 Machine Learning Pipeline (Overview)

The ML pipeline is implemented in **`ml_pipeline.py`** and performs the following steps:

### ✔ **Data Loading**
- Loads Kaggle dataset (`creditcard.csv`)
- If the dataset is unavailable:
  - Downloads it from a public GitHub source
  - OR auto-generates a realistic synthetic dataset

---

### ✔ **Feature Engineering**
- Transaction Amount  
- Time features (Hour, Day, Month)  
- Customer Age  
- Gender Encoding  
- Geo-location features (Latitude & Longitude)  
- Distance calculation between user and merchant  
- Ensures all required model features exist before training

---

### ✔ **Handling Class Imbalance**
- **SMOTE** oversampling for fraudulent classes  
- **Random Under-Sampling** to balance majority/minority classes  

---

### ✔ **Training Models**
Three machine learning models are trained:

1. **Logistic Regression**  
2. **Random Forest**  
3. **Gradient Boosting**

These three are combined into a **weighted ensemble** for better accuracy and robustness.

---

### ✔ **Prediction Outputs**
The system generates:

- Final Prediction → **Fraud / Legitimate**  
- **Fraud Probability**  
- **Confidence Score**  
- **Risk Level** → Low / Medium / High  
- **Model-wise probability scores**  
- **Explainable AI insights** on key contributing factors  

---

<img width="1916" height="1032" alt="image" src="https://github.com/user-attachments/assets/171f9392-1223-4813-80f1-34cf2b5a835d" />
<img width="1915" height="1028" alt="image" src="https://github.com/user-attachments/assets/82da4120-5203-4cf4-bff0-8e2d02a51216" />




