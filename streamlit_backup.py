import streamlit as st

st.title("Face Mask Detection")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image")
    st.success("Upload Successful")