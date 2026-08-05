import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="Face Emotion Recognition",
    page_icon="🎭",
    layout="centered"
)

st.title("🎭 Face Emotion Recognition")
st.write("Upload an image to predict the facial emotion using the trained EfficientNetV2S model.")

# 2. Define Emotion Classes
# Update this list to match your dataset's class folder names if different
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 3. Load Trained Model
@st.cache_resource
def load_emotion_model():
    return tf.keras.models.load_model('best_model.keras')

try:
    model = load_emotion_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Please ensure 'best_model.keras' is in the same directory as this Streamlit app.")
    st.stop()

# 4. Image Preprocessing Function
def preprocess_image(image: Image.Image):
    # Convert image to RGB (in case of RGBA/Grayscale)
    image = image.convert('RGB')
    # Resize image to match model input shape
    image = image.resize((224, 224))
    # Convert to array and expand dimensions to create batch size of 1
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 5. File Upload Interface
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze Emotion"):
        with st.spinner("Processing image..."):
            # Preprocess and predict
            processed_img = preprocess_image(image)
            predictions = model.predict(processed_img)
            
            # Extract results
            predicted_class_idx = np.argmax(predictions[0])
            predicted_label = CLASS_NAMES[predicted_class_idx]
            confidence = predictions[0][predicted_class_idx] * 100

        # Output Results
        st.markdown(f"### Predicted Emotion: **{predicted_label.title()}**")
        st.markdown(f"**Confidence:** `{confidence:.2f}%`")

        # Visual Breakdown of Probabilities
        st.write("---")
        st.subheader("Class Probabilities")
        for i, class_name in enumerate(CLASS_NAMES):
            prob = float(predictions[0][i])
            st.write(f"**{class_name.title()}**")
            st.progress(prob)