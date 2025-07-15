# 📊 Anomaly Detection on Framingham Dataset using MLOps

This project implements anomaly detection using Isolation Forest and Local Outlier Factor (LOF) models on the Framingham Heart Study dataset. The system is deployed with a FastAPI backend and includes training, drift detection, retraining, inference, and API testing pipelines.

---

## 🚀 Features
- Built 2 anomaly models: **Isolation Forest** & **Local Outlier Factor (LOF)**
- Full **MLOps pipeline**: Training, Drift Detection, Retraining, Inference
- Handled:
  - Null Values
  - Categorical Variables
  - Outliers
  - Class Imbalance
  - Feature Engineering & Scaling
- API testing via **Postman** using the best-performing model (Isolation Forest)

---

## 📆 Project Structure
<pre><code>
├── IsolationForestModel/
│   ├── backend/
│   │   ├── main.py (FastAPI app)
│   │   ├── scaler.pkl
│   │   └── 1isolation_forest_model.pkl
├── localOutlierFactorModel/
├── README.md
├── requirements.txt
</code></pre>

## 🤝 How to Run
### 1. Create and Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows

**2. Install Requirements**
pip install -r requirements.txt

**3. Run FastAPI app**
cd IsolationForestModel/backend
uvicorn main:app --reload

**4. Test on Postman**

URL: http://127.0.0.1:8000/detect

Method: POST

Body: raw JSON

{
  "male": 1,
  "age": 55,
  "education": 4,
  "cigsPerDay": 0,
  "BPMeds": 0,
  "prevalentStroke": 0,
  "prevalentHyp": 1,
  "diabetes": 0,
  "totChol": 233,
  "sysBP": 130,
  "diaBP": 85,
  "BMI": 28.5,
  "glucose": 103
}

**📆 Datasets**

framingham_3000.csv used for training

framingham_1240_NT.csv used for drift testing
framingham_3000_NT.csv used for drift testing

framingham.csv used for retraining (framingham_3000.csv + framingham_1240.csv)

3000.csv -> train
3000_NT.csv, 1240_NT.csv -> drift detection
3000.csv + 1240.csv -> retrain

**🌟 Author**
Mounika Reddy Boggari
LinkedIn: linkedin.com/in/mounika-reddy-boggari-a5851b296

