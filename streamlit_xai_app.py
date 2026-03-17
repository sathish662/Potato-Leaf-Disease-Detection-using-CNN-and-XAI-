import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from xai_gradcam import GradCAM, preprocess_image, overlay_heatmap
import matplotlib.pyplot as plt
import cv2

st.set_page_config(page_title="Potato Disease XAI", layout="wide")

@st.cache_resource
def load_model_cached():
    try:
        return load_model('potato_leaf_disease_model.h5')
    except:
        st.error("Model file not found. Please ensure 'potato_leaf_disease_model.h5' is in the repository.")
        st.stop()

model = load_model_cached()
IMG_SIZE = 50
class_names = ['Early_Blight', 'Healthy', 'Late_Blight']

def main():
    st.title("🥔 Potato Disease Detection with Explainable AI")
    st.markdown("Upload a potato leaf image to get disease prediction with Grad-CAM visualization")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Predict & Explain"):
                with st.spinner("Analyzing image..."):
                    try:
                        temp_path = "temp_image.jpg"
                        img.save(temp_path)
                        
                        img_array, original_img = preprocess_image(temp_path, IMG_SIZE)
                        predictions = model.predict(img_array)
                        predicted_class_idx = np.argmax(predictions[0])
                        confidence = np.max(predictions[0])
                        
                        gradcam = GradCAM(model, predicted_class_idx)
                        heatmap = gradcam.compute_heatmap(img_array)
                        overlay, heatmap_colored = overlay_heatmap(heatmap, original_img)
                        
                        st.session_state['predictions'] = predictions
                        st.session_state['predicted_class'] = class_names[predicted_class_idx]
                        st.session_state['confidence'] = confidence
                        st.session_state['overlay'] = overlay
                        st.session_state['heatmap'] = heatmap_colored
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with col2:
        if 'predictions' in st.session_state:
            st.subheader("Prediction Results")
            
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Predicted Class", st.session_state['predicted_class'])
            with col2b:
                st.metric("Confidence", f"{st.session_state['confidence']:.2%}")
            
            st.subheader("Class Probabilities")
            prob_data = {
                class_names[i]: float(st.session_state['predictions'][0][i]) 
                for i in range(len(class_names))
            }
            st.bar_chart(prob_data)
            
            st.subheader("Grad-CAM Visualization")
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(original_img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(st.session_state['heatmap'])
            axes[1].set_title('Heatmap')
            axes[1].axis('off')
            
            axes[2].imshow(st.session_state['overlay'])
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.subheader("Explanation")
            st.info("""
            **Grad-CAM (Gradient-weighted Class Activation Mapping)** highlights the regions 
            in the image that were most important for the model's prediction. 
            Red/yellow areas indicate higher importance, while blue areas indicate lower importance.
            """)

if __name__ == "__main__":
    main()
