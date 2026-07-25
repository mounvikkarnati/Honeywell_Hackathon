# 🛡️ Honeywell Hackathon – AI-Powered Behavioral Anomaly Detection

An AI-powered cybersecurity platform that learns normal user and device behavior, detects anomalies, classifies cyber attacks, and provides explainable risk insights through an interactive SOC dashboard.

## 🚀 Features

- Behavioral Profiling
- Isolation Forest Anomaly Detection
- Random Forest Attack Classification
- SHAP-based Explainable AI
- Real-Time SOC Dashboard
- Automated Evaluation & Report Generation

## 🛠️ Tech Stack

**Python • FastAPI • React • PostgreSQL • Scikit-learn • Pandas • SHAP • Plotly**

## 📂 Project Structure

```text
backend/
frontend/
generator/
ml/
evaluation/
models/
output/
```

## ▶️ Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the complete pipeline:

```bash
python -m run_pipeline
```

Start the dashboard:

```bash
uvicorn backend.app:app --reload
```

Open:

- Dashboard: http://localhost:8000/
- API Docs: http://localhost:8000/docs

## 📌 Highlights

- 109K+ Synthetic Security Events
- 330 Behavioral Entities
- 7 Simulated Attack Types
- Explainable AI Alerts
- End-to-End Automated Pipeline

## 👨‍💻 Author

**Mounvik Karnati**  
VIT-AP University
