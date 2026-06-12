# Face Mask Detection System

A deep learning-based Face Mask Detection application that identifies whether a person is wearing a face mask using both uploaded images and real-time webcam streams. The project uses OpenCV for face detection, TensorFlow/Keras for classification, and Streamlit for deployment.

## Live Applications

### Image Upload Detection

https://face-mask-detection-sobtc646yq4rrjkvevpax5.streamlit.app/

### Live Webcam Detection

https://face-mask-detection-fvfhn49dmxaytbdouwujzg.streamlit.app/

## Repository

https://github.com/abinavdev/face-mask-detection

---

## Overview

This project combines computer vision and deep learning techniques to detect faces and classify each detected face as either:

* Mask
* No Mask

The system supports both:

* Image Upload Detection
* Real-Time Webcam Detection

It can detect multiple faces simultaneously and provides confidence scores for every prediction.

---

## Features

### Image Upload Detection

* Upload image files for analysis
* Face detection using OpenCV DNN
* Mask / No Mask classification
* Multiple face detection support
* Confidence score display
* Download processed image

### Live Webcam Detection

* Browser webcam access
* Real-time face detection
* Real-time mask classification
* Confidence score display
* Multiple face detection support
* Streamlit WebRTC integration

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

### Supporting Libraries

* NumPy
* Pillow
* Imutils
* AV

---

## System Workflow

### Image Upload Pipeline

1. User uploads an image.
2. OpenCV detects faces within the image.
3. Each detected face is extracted and preprocessed.
4. MobileNetV2 performs mask classification.
5. The system labels each face as:

   * Mask
   * No Mask
6. The processed image is displayed with predictions.

### Live Webcam Pipeline

1. User grants webcam access.
2. Browser video stream is captured through Streamlit WebRTC.
3. OpenCV performs face detection on each frame.
4. MobileNetV2 classifies detected faces.
5. Results are displayed in real-time with confidence scores.

---

## Project Structure

```text
face-mask-detection/
│
├── streamlit_app.py
├── streamlit_live.py
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

## Run Image Upload Version

```bash
streamlit run streamlit_app.py
```

---

## Run Live Webcam Version

```bash
streamlit run streamlit_live.py
```

---

## Model Information

| Parameter | Value                    |
| --------- | ------------------------ |
| Model     | MobileNetV2              |
| Framework | TensorFlow / Keras       |
| Task      | Face Mask Classification |
| Classes   | Mask, No Mask            |

---

## Sample Use Cases

* Public safety monitoring
* Educational demonstrations of computer vision
* Deep learning portfolio project
* Face mask compliance analysis
* Real-time webcam-based detection systems

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
