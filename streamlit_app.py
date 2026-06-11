import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

st.title("Face Mask Detection")

# Load face detector
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load mask detector model
maskNet = load_model("mask_detector.h5")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)
    image = np.array(image)

    orig = image.copy()
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image,
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    faceNet.setInput(blob)
    detections = faceNet.forward()

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

            face = image[startY:endY, startX:endX]

            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))

            face = img_to_array(face)
            face = preprocess_input(face)

            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = maskNet.predict(
                face,
                verbose=0
            )[0]

            label = (
                "Mask"
                if mask > withoutMask
                else "No Mask"
            )

            color = (
                (0, 255, 0)
                if label == "Mask"
                else (0, 0, 255)
            )

            label_text = (
                f"{label}: "
                f"{max(mask, withoutMask) * 100:.2f}%"
            )

            cv2.putText(
                orig,
                label_text,
                (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            cv2.rectangle(
                orig,
                (startX, startY),
                (endX, endY),
                color,
                2
            )

    st.image(
        orig,
        channels="BGR",
        caption="Prediction Result"
    )