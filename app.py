from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import tempfile
from xai_gradcam import GradCAM, preprocess_image, overlay_heatmap
import base64
from PIL import Image
import io

app = Flask(__name__)

# Load model
model = load_model('potato_leaf_disease_model.h5')
IMG_SIZE = 50
class_names = ['Early_Blight', 'Healthy', 'Late_Blight']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and process image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # Save temporarily for Grad-CAM processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name
        
        # Preprocess for prediction
        img_array, original_img = preprocess_image(tmp_path, IMG_SIZE)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        
        # Generate Grad-CAM
        gradcam = GradCAM(model, predicted_class_idx)
        heatmap = gradcam.compute_heatmap(img_array)
        overlay, heatmap_colored = overlay_heatmap(heatmap, original_img)
        
        # Convert images to base64
        def img_to_base64(img):
            buffer = io.BytesIO()
            if isinstance(img, np.ndarray):
                img_pil = Image.fromarray(img)
            else:
                img_pil = img
            img_pil.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode()
        
        original_b64 = img_to_base64(original_img)
        heatmap_b64 = img_to_base64(heatmap_colored)
        overlay_b64 = img_to_base64(overlay)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Prepare class probabilities
        class_probs = {
            class_names[i]: float(predictions[0][i]) 
            for i in range(len(class_names))
        }
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': float(confidence),
            'class_probabilities': class_probs,
            'original_image': f'data:image/jpeg;base64,{original_b64}',
            'heatmap': f'data:image/jpeg;base64,{heatmap_b64}',
            'overlay': f'data:image/jpeg;base64,{overlay_b64}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
