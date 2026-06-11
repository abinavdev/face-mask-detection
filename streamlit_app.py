import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Face Mask Detection Dashboard",
    layout="wide"
)

# --------------------------------------------------
# MODEL LOADING
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
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("Project Info")

    st.markdown("### Tech Stack")

    st.write("• Python")
    st.write("• TensorFlow")
    st.write("• Keras")
    st.write("• OpenCV")
    st.write("• MobileNetV2")
    st.write("• Streamlit")

    st.divider()

    st.markdown("### Model")

    st.write("Face Detection: OpenCV DNN")
    st.write("Mask Classification: MobileNetV2")

    st.divider()

    st.markdown("### About")

    st.write(
        "This application detects whether a person "
        "is wearing a face mask from uploaded images."
    )

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("Face Mask Detection Dashboard")

st.markdown(
    """
    Upload an image and detect whether people are wearing
    face masks using a Deep Learning model built with
    TensorFlow, Keras, MobileNetV2, and OpenCV.
    """
)

st.divider()

# --------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------

left_col, right_col = st.columns([2, 1])

with left_col:

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

with right_col:

    st.info(
        """
        **How it works**

        1. Upload an image
        2. Faces are detected using OpenCV DNN
        3. Each face is classified
        4. Results are displayed with confidence scores
        """
    )

# --------------------------------------------------
# DETECTION
# --------------------------------------------------

if uploaded_file is not None:

    try:

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

        total_faces = 0
        mask_count = 0
        no_mask_count = 0

        results = []

        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:

                total_faces += 1

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

                face = np.expand_dims(
                    face,
                    axis=0
                )

                (mask, withoutMask) = maskNet.predict(
                    face,
                    verbose=0
                )[0]

                label = (
                    "Mask"
                    if mask > withoutMask
                    else "No Mask"
                )

                confidence_score = max(
                    mask,
                    withoutMask
                ) * 100

                if label == "Mask":
                    color = (0, 255, 0)
                    mask_count += 1
                else:
                    color = (0, 0, 255)
                    no_mask_count += 1

                label_text = (
                    f"{label} "
                    f"({confidence_score:.1f}%)"
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

                results.append(
                    {
                        "Face": total_faces,
                        "Prediction": label,
                        "Confidence": f"{confidence_score:.2f}%"
                    }
                )

        # ------------------------------------------
        # STATISTICS
        # ------------------------------------------

        st.subheader("Detection Statistics")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Faces",
            total_faces
        )

        c2.metric(
            "Mask",
            mask_count
        )

        c3.metric(
            "No Mask",
            no_mask_count
        )

        st.divider()

        # ------------------------------------------
        # IMAGE RESULT
        # ------------------------------------------

        st.subheader("Prediction Result")

        st.image(
            orig,
            channels="BGR",
            use_container_width=True
        )

        st.divider()

        # ------------------------------------------
        # SUMMARY TABLE
        # ------------------------------------------

        if results:

            st.subheader(
                "Detection Summary"
            )

            df = pd.DataFrame(results)

            st.dataframe(
                df,
                use_container_width=True
            )

        # ------------------------------------------
        # DOWNLOAD BUTTON
        # ------------------------------------------

        success, buffer = cv2.imencode(
            ".jpg",
            orig
        )

        if success:

            st.download_button(
                label="Download Processed Image",
                data=buffer.tobytes(),
                file_name="mask_detection_result.jpg",
                mime="image/jpeg"
            )

    except Exception as e:

        st.error(
            f"Error processing image: {e}"
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style='text-align:center'>
        <p>
            Face Mask Detection System |
            Built with TensorFlow, OpenCV and Streamlit
        </p>
    </div>
    """,
    unsafe_allow_html=True
)