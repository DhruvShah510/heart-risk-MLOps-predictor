## Heart Disease Risk Prediction System (MLOps)
A full-stack **MLOps project** that predicts the risk of heart disease based on clinical patient data. This system is designed for **medical professionals**, offering not just a prediction but also a **confidence score** to aid in clinical interpretation.

🚀 Live Demo
Live project link : https://heart-risk-predictor-29lb.onrender.com

The above is the live project link (The project is deployed on render)
> **Note:** As this application is hosted on a free tier, the first load may take **30–60 seconds** to spin up.

---

## 🧠 Project Overview

Predicting heart disease risk is a complex task requiring the analysis of multiple physiological factors.  
This project implements a **Random Forest Classifier pipeline** to analyze **13 clinical features** and provide real-time predictions.

## ✨ Key Features

- **Clinical Interpretability**  
  Designed for healthcare scenarios, requiring detailed medical inputs such as cholesterol, thalach, oldpeak, etc.

- **Real-time Inference**  
  High-performance backend built using **FastAPI**.

- **Interactive UI**  
  Clean and professional frontend developed with **Streamlit**.

- **Containerized Architecture**  
  Fully dockerized for **plug-and-play deployment**.



## 🛠️ Tech Stack

- **Machine Learning:** Scikit-learn, Pandas, NumPy  
- **Backend API:** FastAPI, Uvicorn  
- **Frontend UI:** Streamlit  
- **DevOps / Deployment:** Docker, Docker Hub, Render  
- **Environment & Scripting:** Shell Scripting (Linux / Bash)



## 📂 Project Structure

```text
.
├── fastapi_part/          # Backend API logic
│   └── main.py            # FastAPI application routes
├── frontend/              # Streamlit frontend
│   └── frontend.py        # UI logic and API communication
├── rf_pipeline.joblib     # Pre-trained ML model artifact
├── start.sh               # Shell script to orchestrate dual services
├── Dockerfile             # Container configuration
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```



## ⚙️ How It Works (Architecture)

1. **Model Loading**  
   The FastAPI server loads the `rf_pipeline.joblib` model artifact during application startup.

2. **REST API**  
   The backend exposes a `/predict` endpoint that accepts patient data via **POST requests**.

3. **Frontend Interaction**  
   The Streamlit application collects user inputs and communicates with the FastAPI backend through internal networking.

4. **Service Orchestration**  
   A `start.sh` script runs both FastAPI and Streamlit concurrently within a **single Docker container**, optimized for cloud platforms like **Render**.



## 🐳 Local Setup & Docker

1. Pull the Image from Docker Hub

```bash
docker pull dhruvshah15/heart-risk-predictor:latest
```

2. Run the Container:
```bash
docker run -p 8501:8501 dhruvshah15/heart-risk-predictor:latest
```

3. Access the App: Open your browser and go to
```bash
http://localhost:8501.
```

## 🤝 Contact
Dhruv Shah

Linked in : https://www.linkedin.com/in/dhruv-shah-25997624b
