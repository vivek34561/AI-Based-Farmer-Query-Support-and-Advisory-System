import streamlit as st
import requests
import os
import uuid
from dotenv import load_dotenv
load_dotenv()
st.set_page_config(page_title="🌱 AI Farmer Advisory System", layout="wide")

# Sidebar options
st.sidebar.header("Settings")
city_choice = st.sidebar.text_input("🌍 Enter your city (optional)")
lang_choice = st.sidebar.selectbox("🌐 Choose Response Language", ["en", "ml", "hi", "ta"])

tab1, tab2 = st.tabs(["🌾 Crop Disease Prediction", "🤖 AI Chatbot Advisor"])

# --- Tab 1: Image Prediction ---
with tab1:
    st.header("Upload Crop Image for Disease Prediction")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Open the file for sending inside a 'with' block
        with open(file_path, "rb") as f:
            files = {"file": f}
            response = requests.post("http://127.0.0.1:8000/api/predict-disease/", files=files)

        if response.status_code == 200:
            data = response.json()
            st.image(file_path, caption="Uploaded Image", use_container_width=True)
            st.subheader("Predicted Disease:")
            st.write(data.get("prediction", "No prediction available"))
            st.write("Probabilities:", data.get("probabilities", []))
        else:
            st.error("Error in prediction")

        # Now the file is closed and can be removed safely
        os.remove(file_path)


# --- Tab 2: Chatbot ---


with tab2:
    st.header("AI Farming Advice Chatbot")
    
    # Generate a session_id per user to resume chat
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    user_query = st.text_area("Ask your farming question:")

    if st.button("Get Advice") and user_query:
        payload = {
            "session_id": st.session_state.session_id,
            "query": user_query,
            "lang": "en"  # You can let user choose
        }

        response = requests.post("http://127.0.0.1:8000/api/farmer-query/", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            advice = data.get("advice", "No advice available")
            
            # Streaming word by word
            placeholder = st.empty()
            words = advice.split()
            display_text = ""
            for word in words:
                display_text += word + " "
                placeholder.text(display_text)
        else:
            st.error("Error fetching advice")