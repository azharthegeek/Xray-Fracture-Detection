import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-color: #b00734; /* Change background color to red */
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    text-align: center; /* Center the title */
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Function to preprocess the image based on model requirements
def preprocess_image(img, target_shape):
    img = img.convert("RGB")  
    img = np.array(img)
    target_dim = (224, 224) 
    img = cv2.resize(img, target_dim, interpolation=cv2.INTER_AREA)
    img = img / 255.0 
    img = np.expand_dims(img, axis=0)
    return img

st.title("TechShoor MediScan")
st.title("Fracture Detection")
st.write("Welcome to the Fracture Detection App!")
st.write("Unlocking Medical Imaging Breakthroughs")

# Allow user to select model
model_names_mapping = {
    "Elbow": "elbow_converted_model.tflite",
    "Finger": "finger_converted_model.tflite",
    "Forearm": "foream_converted_model.tflite",
    "Hand": "hand_converted_model.tflite",
    "Humerus": "humerus_converted_model.tflite",
    "Shoulder": "shoulder_converted_model.tflite",
    "Wrist": "wrist_converted_model.tflite"
}

# Allow user to select model
selected_model_name = st.selectbox("Select Model", list(model_names_mapping.keys()))

# Get the actual file name corresponding to the selected model name
selected_model = model_names_mapping[selected_model_name]

# Allow user to upload image
uploaded_file = st.file_uploader("Upload an X-ray image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

# Predict button
if st.button("Predict"):
    if uploaded_file is not None:
        # Load TFLite model
        try:
            interpreter = tf.lite.Interpreter(model_path=selected_model)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            st.write("Model loaded successfully.")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()

        # Display uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        try:
            model_input_shape = input_details[0]['shape'][1:3] 
            preprocessed_img = preprocess_image(img, model_input_shape)

            # Convert input data to FLOAT32
            preprocessed_img = preprocessed_img.astype(np.float32)

            # Get predictions
            interpreter.set_tensor(input_details[0]['index'], preprocessed_img)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])
            confidence = predictions[0][0]  # Assuming only binary classification
            
            # Set a threshold for uncertainty
            uncertainty_threshold = 0.7

            # Handle uncertain predictions
            if confidence < uncertainty_threshold:
                st.write("**Uncertain Prediction:** The model's predicted accuracy is low. Consider uploading additional images from different angles for better assessment.")
            else:
                predicted_label = "Fractured" if confidence > 0.5 else "Not Fractured"
                st.write("**Prediction:**", predicted_label)
                st.write("**Confidence Score:** {:.2%}".format(confidence))

                st.write("**Predicted Label:**", "Positive (Abnormal/Fractured)" if predicted_label == "Fractured" else "Negative (Normal/Not Fractured)")
                st.write("**Predicted Accuracy:** {:.2f}%".format(confidence * 100))

                st.write("Prediction is likely reliable.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
