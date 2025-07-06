# 🎯 AI Attention Detection System

An intelligent real-time attention monitoring system that uses computer vision and machine learning to detect and predict attention levels through facial analysis, hand detection, and phone usage monitoring.

## 📋 Table of Contents
- [Features](#-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset Format](#-dataset-format)
- [Model Training](#-model-training)
- [System Architecture](#-system-architecture)
- [Controls](#-controls)
- [Output Files](#-output-files)
- [Collaborators](#-collaborators)

## ✨ Features

### 🔍 **Real-time Detection**
- **Face Detection**: Detects faces using Haar Cascade classifiers
- **Hand Detection**: Uses YOLOv8 pose estimation with fallback to skin color detection
- **Phone Detection**: YOLOv8 object detection with geometric fallback analysis
- **Pose Estimation**: Determines head orientation (Forward, Left, Right, Down)

### 🧠 **Machine Learning Prediction**
- **10-second Data Collection**: Automatically collects detection data every second
- **Feature Averaging**: Calculates statistical averages of detection features
- **RandomForest Classification**: Predicts attention level (Attentive/Inattentive)
- **Confidence Scoring**: Provides prediction confidence percentages

### 📊 **Data Management**
- **JSON Export**: Saves detection data and predictions in structured format
- **CSV Training**: Supports custom dataset training
- **Real-time Monitoring**: Live display of detection statistics

## 🎥 Demo

![Attention Detection Demo](demo.gif)

*The system provides real-time feedback on attention levels with visual overlays showing detected faces, hands, and phones.*

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam/Camera
- macOS, Windows, or Linux

### Step 1: Clone Repository
```bash
git clone https://github.com/zyaaa-aaa/attention_tracker.git
cd attention-tracker
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Dataset
Ensure that the training dataset `attention_detection_dataset_v1.csv` is in the project directory.

## 🎮 Usage

### Basic Usage
```bash
python main.py
```

### Training Custom Model
1. Prepare your dataset in the correct format (see [Dataset Format](#dataset-format))
2. Place the CSV file as `attention_detection_dataset_v1.csv`
3. Run the system - it will automatically train the model on startup

### Real-time Prediction
1. Start the system
2. Wait for the camera to initialize
3. Press **'r'** to start 10-second attention prediction
4. System will collect data and provide prediction results

## 📊 Dataset Format

Your training dataset should be a CSV file with the following columns:

| Column | Description | Type | Example |
|--------|-------------|------|---------|
| `no_of_face` | Number of faces detected | Integer | 1 |
| `face_x` | X-coordinate (upper-left corner) | Integer | 320 |
| `face_y` | Y-coordinate (upper-left corner) | Integer | 240 |
| `face_w` | Face width in pixels | Integer | 150 |
| `face_h` | Face height in pixels | Integer | 180 |
| `face_con` | Face detection confidence | Float | 0.85 |
| `no_of_hand` | Number of hands detected | Integer | 2 |
| `pose` | Head orientation | String | "Forward" |
| `pose_x` | X-axis rotation angle | Integer | 5 |
| `pose_y` | Y-axis rotation angle | Integer | -2 |
| `phone` | Phone detection (0/1) | Integer | 0 |
| `phone_x` | Phone X-coordinate | Integer | 100 |
| `phone_y` | Phone Y-coordinate | Integer | 300 |
| `phone_w` | Phone width | Integer | 80 |
| `phone_h` | Phone height | Integer | 160 |
| `phone_con` | Phone detection confidence | Float | 0.72 |
| `label` | **Target**: 0=Attentive, 1=Inattentive | Integer | 0 |

### Sample Dataset Row
```csv
no_of_face,face_x,face_y,face_w,face_h,face_con,no_of_hand,pose,pose_x,pose_y,phone,phone_x,phone_y,phone_w,phone_h,phone_con,label
1,320,240,150,180,0.85,2,Forward,5,-2,0,0,0,0,0,0.0,0
```

## 🧠 Model Training

The system uses a **RandomForest Classifier** with the following pipeline:
- **Preprocessing**: OneHotEncoder for categorical features, passthrough for numerical
- **Model**: RandomForest with 100 estimators, max_depth=10
- **Evaluation**: ROC-AUC score, precision, recall, and accuracy metrics

### Training Process
```python
# Automatic training on startup
detector = CameraDetectionSystem()
# Model trains automatically when system initializes
```

### Model Performance
The system displays training metrics including:
- ROC-AUC Score
- Accuracy
- Precision and Recall for both classes

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Camera Input  │───▶│  Detection Layer │───▶│ Feature Extract │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Prediction UI   │◀───│   ML Pipeline    │◀───│ Data Collection │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Detection Components
- **Face Detection**: Haar Cascade (OpenCV)
- **Hand Detection**: YOLOv8 Pose → Skin Color Fallback
- **Phone Detection**: YOLOv8 Object Detection → Geometric Fallback
- **Pose Estimation**: Geometric analysis based on face position

## 🎮 Controls

| Key | Action |
|-----|--------|
| **SPACE** | Save current detection data |
| **'p'** | Print current detection data to console |
| **'r'** | Start 10-second attention prediction |
| **'s'** | Stop ongoing data collection |
| **'h'** | Toggle hand detection method |
| **'q'** | Quit application |

## 📁 Output Files

### Detection Data
```json
{
  "timestamp": "2024-01-15T14:30:25.123456",
  "frame_data": {
    "no_of_face": 1,
    "face_x": 320,
    "face_y": 240,
    ...
  }
}
```

### Prediction Results
```json
{
  "timestamp": "2024-01-15T14:30:25.123456",
  "averaged_features": { ... },
  "prediction": 0,
  "prediction_meaning": "1=Inattentive, 0=Attentive",
  "probability_inattentive": 0.25,
  "probability_attentive": 0.75,
  "attention_status": "ATTENTIVE",
  "confidence_percentage": "75.0%",
  "data_points_collected": 10
}
```

## 👥 Collaborators

This project was developed as part of a collaborative effort:

### 🎓 Team

| Name | Institution |
|------|-------------|
| **Shazya Audrea Taufik** | Institut Teknologi Bandung | 
| **Tamara Mayranda Lubis** | Institut Teknologi Bandung | 
| **Yovanka Sandrina Maharaja** | Institut Teknologi Bandung | 

---

<div align="center">

**⭐ Thank You! ⭐**

</div>