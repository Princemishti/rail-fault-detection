

---
title: Railway Track Fault Detection System
emoji: 🛤️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Smart Solar Rail Track Fault Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-green.svg)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688.svg)](https://fastapi.tiangolo.com/)

An advanced AI-powered system designed for real-time monitoring and fault detection in railway tracks. This project integrates Computer Vision (YOLOv8), IoT simulation, and a web-based dashboard to provide an end-to-end solution for railway safety, powered by simulated solar energy.

## 🚀 Key Features

- **Real-time AI Classification**: Utilizes a custom-trained YOLOv8 model to classify rail tracks as "Defective" or "Non-Defective" with high accuracy.
- **IoT Edge Simulation**: Mimics a solar-powered IoT node that captures images and transmits them to a central server for analysis.
- **Interactive Dashboard**: A modern web interface to visualize live analysis, view fault alerts, and track historical data.
- **Energy-Aware Simulation**: Includes a simulated battery management system, reflecting the solar-powered nature of the edge sensors.
- **Comprehensive Analytics**: Generates detailed metrics, including accuracy curves, confusion matrices, and loss plots.

## 📁 Project Structure

| File/Folder | Description |
| :--- | :--- |
| `main.py` | Core FastAPI backend server and API endpoints. |
| `sensor_sim.py` | IoT sensor simulation script for edge node behavior. |
| `train_model.py` | Script for training the YOLOv8 classification model. |
| `evaluate_model.py`| Performance evaluation and metric generation. |
| `static/` | UI assets, generated charts, and processed images. |
| `templates/` | HTML templates for the web dashboard. |
| `requirements.txt`| Project dependencies. |

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Git

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/shreyash-jain27/INT-345-Project.git
   cd INT-345-Project
   ```

2. **Create and Activate Virtual Environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Dataset (Optional)**
   The project expects data for training. Use the provided script:
   ```bash
   python download_data.py
   ```

## 🚦 Usage

1. **Start the Backend Server**
   ```bash
   python main.py
   ```
   Open your browser and navigate to `http://localhost:8000`.

2. **Run the Sensor Simulation**
   In a separate terminal, start the IoT simulation:
   ```bash
   python sensor_sim.py
   ```

3. **View Results**
   Monitor the web dashboard for real-time classifications and fault detection logs.

## 📊 Model Training & Evaluation
To retrain the model or evaluate its performance:
- Use `train_model.py` for training on your dataset.
- Use `evaluate_model.py` and `create_charts.py` to generate performance visualizations.

## 🚀 Cloud Deployment

### 1. Hugging Face Spaces (Free & Recommended)
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Select **Docker** as the Space SDK and **Free CPU**.
3. Push or upload this repository to your Space.
4. Hugging Face automatically reads the `Dockerfile` and deploys the app on port `7860`.

### 2. Render.com
1. Create a new **Web Service** on [Render](https://render.com) and link your GitHub repository.
2. Render detects `render.yaml` or selects **Docker** runtime automatically.
3. Deploy for free with automatic SSL.

### 3. Railway.app
1. Create a new project on [Railway](https://railway.app) and select **Deploy from GitHub repo**.
2. Railway detects `railway.json` and builds via `Dockerfile`.

### 4. Local Docker
```bash
docker build -t railway-detection .
docker run -p 7860:7860 railway-detection
```

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Acknowledgments
- Dataset provided by [Salman Ibne Eunus](https://www.kaggle.com/datasets/salmaneunus/railway-track-fault-detection) on Kaggle.
- YOLOv8 by [Ultralytics](https://github.com/ultralytics/ultralytics).
