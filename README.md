# Face Mask Detection System

A production-ready AI-powered Face Mask Detection System built using Deep Learning, Computer Vision, and Streamlit.

The application detects whether individuals are wearing face masks from uploaded images and live webcam streams using OpenCV and a MobileNetV2-based TensorFlow model. The system supports multiple face detection, real-time monitoring, interactive analytics, and a modern web-based interface.

---

## Live Demo

https://face-mask-detection-ipzcpqfzegbxzowxghz8xx.streamlit.app/

---

## Repository

https://github.com/abinavdev/face-mask-detection

---

## Overview

This project combines Computer Vision and Deep Learning techniques to detect faces and classify each detected face as either:

* Mask
* No Mask

The application supports:

* Image Upload Detection
* Real-Time Webcam Detection
* Interactive Analytics Dashboard
* System Monitoring Dashboard

The system can detect multiple faces simultaneously and provides confidence scores for every prediction.

---

## Key Highlights

* Developed a Deep Learning-based Face Mask Detection System using MobileNetV2 and TensorFlow.
* Implemented face detection using OpenCV DNN and confidence-based mask classification.
* Built a modern multi-page Streamlit application with dashboard, analytics, image upload, and live webcam detection.
* Integrated browser webcam streaming using Streamlit WebRTC.
* Created interactive analytics dashboards using Plotly.
* Deployed the application on Streamlit Cloud.
* Managed version control and deployment using Git and GitHub.

---

## Features

### Dashboard

* Modern AI-inspired user interface
* Real-time platform status monitoring
* Detection statistics overview
* Model information panel
* System health indicators
* Interactive analytics widgets

### Image Upload Detection

* Drag-and-drop image upload
* Face detection using OpenCV DNN
* Mask / No Mask classification
* Multiple face detection support
* Confidence score visualization
* Detection summary statistics
* Processed image preview
* Download processed image

### Live Webcam Detection

* Browser webcam access
* Real-time face detection
* Real-time mask classification
* Confidence score display
* Multiple face detection support
* Streamlit WebRTC integration
* Live monitoring interface

### Analytics

* Detection statistics dashboard
* Interactive Plotly visualizations
* Compliance metrics monitoring
* Detection distribution charts
* Performance monitoring widgets

---

## Technology Stack

### Programming Language

* Python

### Machine Learning

* TensorFlow
* Keras
* MobileNetV2

### Computer Vision

* OpenCV
* OpenCV DNN Face Detector

### Web Application

* Streamlit
* Streamlit WebRTC
* Plotly
* Streamlit Option Menu

### Supporting Libraries

* NumPy
* Pandas
* Pillow
* Imutils
* AV

---

## System Architecture

```text
Image / Webcam Input
          ↓
Face Detection (OpenCV DNN)
          ↓
Face Extraction
          ↓
Image Preprocessing
          ↓
MobileNetV2 Classification
          ↓
Mask / No Mask Prediction
          ↓
Visualization & Analytics
```

---

## Image Upload Workflow

1. User uploads an image.
2. OpenCV detects faces within the image.
3. Each detected face is extracted and preprocessed.
4. MobileNetV2 performs mask classification.
5. The system labels each face as:

   * Mask
   * No Mask
6. Results are displayed with confidence scores and visual overlays.

---

## Live Webcam Workflow

1. User grants webcam access.
2. Browser video stream is captured using Streamlit WebRTC.
3. OpenCV performs face detection on each frame.
4. MobileNetV2 classifies detected faces.
5. Results are displayed in real time with confidence scores.

---

## Model Information

| Parameter | Value                    |
| --------- | ------------------------ |
| Model     | MobileNetV2              |
| Framework | TensorFlow / Keras       |
| Task      | Face Mask Classification |
| Classes   | Mask, No Mask            |

---

## Project Structure

```text
face-mask-detection/
│
├── streamlit_app.py
├── app.py
├── detect_mask_video.py
├── mask_detector.h5
├── requirements.txt
├── runtime.txt
│
├── face_detector/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
├── screenshots/
│
├── .streamlit/
│   └── config.toml
│
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/abinavdev/face-mask-detection.git
cd face-mask-detection
```

### Create a Virtual Environment

```bash
conda create -n maskdetect python=3.11
conda activate maskdetect
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run streamlit_app.py
```

---

## Deployment

The application is deployed on Streamlit Cloud.

Live URL:

https://face-mask-detection-ipzcpqfzegbxzowxghz8xx.streamlit.app/

---

## Sample Use Cases

* Public safety monitoring
* Educational demonstrations of Computer Vision
* Deep Learning portfolio project
* Face mask compliance analysis
* Real-time webcam monitoring systems
* AI-powered surveillance applications

---

## Skills Demonstrated

* Deep Learning
* Computer Vision
* Image Processing
* TensorFlow
* Keras
* OpenCV
* MobileNetV2
* Streamlit
* Streamlit WebRTC
* Plotly
* Data Visualization
* Git and GitHub
* Model Deployment
* Web Application Development

---

## Screenshots

### Mask Detection Result 1

![Mask Detection Result 1](screenshots/mask_detection_result%20\(1\).jpg)

### Mask Detection Result 2

![Mask Detection Result 2](screenshots/mask_detection_result%20\(2\).jpg)

### Mask Detection Result 3

![Mask Detection Result 3](screenshots/mask_detection_result%20\(3\).jpg)

---

## Future Enhancements

* Video file analysis
* Detection analytics dashboard
* YOLO-based face detection
* Face recognition integration
* Attendance monitoring system
* Mobile-responsive UI improvements
* Docker containerization

---




LinkedIn: https://linkedin.com/in/abinavdev
