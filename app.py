import streamlit as st
import base64
from src.cnnClassifier.utils.common import decodeImage
from src.cnnClassifier.pipeline.prediction import PredictionPipeline

# Function to convert image to base64 string
def encode_image(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# App title
st.title("🌱 Crop Disease Classification")

st.header("Upload an Image for Prediction")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file
    file_path = "inputImage.jpg"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Encode and decode image
    encoded_image = encode_image(file_path)
    decodeImage(encoded_image, file_path)

    # Run prediction pipeline
    prediction_pipeline = PredictionPipeline(file_path)
    result = prediction_pipeline.predict()

    st.image(file_path, caption="Uploaded Image", use_column_width=True)
    st.subheader("Predicted Disease:")
    st.write(result[0]["image"])

    # Show top 5 predictions
    probs = result[0]["probabilities"]
    class_indices = prediction_pipeline.class_indices
    class_labels = {v: k for k, v in class_indices.items()}

    sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:5]
    st.subheader("Top Predictions:")
    for idx, prob in sorted_probs:
        st.write(f"{class_labels[idx]}: {prob:.4f}")
