# Face Mask Detection System

A deep learning-based Face Mask Detection application that identifies whether a person is wearing a face mask from an uploaded image. The application uses OpenCV for face detection, TensorFlow/Keras for classification, and Streamlit for deployment.

## Live Demo

https://face-mask-detection-sobtc646yq4rrjkvevpax5.streamlit.app/

## Repository

https://github.com/abinavdev/face-mask-detection

---

## Overview

This project combines computer vision and deep learning techniques to detect faces in an image and classify each detected face as either:

- Mask
- No Mask

The application supports both single-face and multi-face detection and provides prediction confidence scores for each detected face.

---

## Features

- Face detection using OpenCV DNN
- Face mask classification using a trained MobileNetV2 model
- Multiple face detection in a single image
- Confidence score display
- Streamlit-based web interface
- Public cloud deployment

---

## Technology Stack

### Programming Language

- Python

### Machine Learning

- TensorFlow
- Keras
- MobileNetV2

### Computer Vision

- OpenCV

### Web Application

- Streamlit

### Supporting Libraries

- NumPy
- Pillow
- Imutils

---

## System Workflow

1. User uploads an image.
2. OpenCV detects faces within the image.
3. Each detected face is extracted and preprocessed.
4. The trained MobileNetV2 model performs mask classification.
5. The system labels each face as:
   - Mask
   - No Mask
6. The processed image with predictions is displayed to the user.

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

### Run the Application

```bash
streamlit run streamlit_app.py
```

---

## Model Information

| Parameter | Value |
|------------|---------|
| Model | MobileNetV2 |
| Framework | TensorFlow / Keras |
| Task | Face Mask Classification |
| Classes | Mask, No Mask |

---

## Sample Use Cases

- Public safety monitoring
- Educational demonstrations of computer vision
- Deep learning portfolio project
- Face mask compliance analysis

---

## Skills Demonstrated

- Deep Learning
- Computer Vision
- Image Processing
- TensorFlow
- Keras
- OpenCV
- Streamlit
- Git and GitHub
- Model Deployment
- Web Application Development

---

## Future Enhancements

- Real-time webcam detection
- Video file analysis
- Improved user interface
- Detection statistics dashboard
- Mobile-responsive design
- Performance optimization

---

## Author

**Abinav A D**

B.Tech Computer Science Engineering Student

GitHub: https://github.com/abinavdev

---

## License

This project is intended for educational, learning, and portfolio purposes.
