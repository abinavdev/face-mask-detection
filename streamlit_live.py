import streamlit as st
import cv2
import av
import numpy as np

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Live Face Mask Detection",
    layout="wide"
)

st.title("Live Face Mask Detection")

st.write(
    "Real-time face mask detection using webcam, "
    "OpenCV DNN and MobileNetV2."
)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    faceNet = cv2.dnn.readNet(
        "face_detector/deploy.prototxt",
        "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    )

    maskNet = load_model("mask_detector.h5")

    return faceNet, maskNet


faceNet, maskNet = load_models()

# --------------------------------------------------
# FACE MASK DETECTION FUNCTION
# --------------------------------------------------

def detect_and_predict_mask(frame):

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame,
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    faceNet.setInput(blob)

    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:

            box = detections[0, 0, i, 3:7] * np.array(
                [w, h, w, h]
            )

            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            face = frame[startY:endY, startX:endX]

            if face.size == 0:
                continue

            face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )

            face = cv2.resize(
                face,
                (224, 224)
            )

            face = img_to_array(face)

            face = preprocess_input(face)

            faces.append(face)

            locs.append(
                (startX, startY, endX, endY)
            )

    if len(faces) > 0:

        faces = np.array(
            faces,
            dtype="float32"
        )

        preds = maskNet.predict(
            faces,
            batch_size=32,
            verbose=0
        )

    return (locs, preds)

# --------------------------------------------------
# VIDEO PROCESSOR
# --------------------------------------------------

class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        (locs, preds) = detect_and_predict_mask(
            img
        )

        for (box, pred) in zip(locs, preds):

            (startX, startY, endX, endY) = box

            (mask, withoutMask) = pred

            label = (
                "Mask"
                if mask > withoutMask
                else "No Mask"
            )

            confidence = (
                max(mask, withoutMask)
                * 100
            )

            color = (
                (0, 255, 0)
                if label == "Mask"
                else (0, 0, 255)
            )

            text = (
                f"{label} "
                f"({confidence:.1f}%)"
            )

            cv2.putText(
                img,
                text,
                (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            cv2.rectangle(
                img,
                (startX, startY),
                (endX, endY),
                color,
                2
            )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

# --------------------------------------------------
# START WEBCAM
# --------------------------------------------------

st.subheader("Webcam Stream")

webrtc_streamer(
    key="mask-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)