import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class GradCAM:
    def __init__(self, model, class_idx, layer_name=None):
        self.model = model
        self.class_idx = class_idx
        self.layer_name = layer_name
        self.grad_model = self._make_grad_model()
    
    def _make_grad_model(self):
        if self.layer_name is None:
            # Find the last convolutional layer
            for layer in reversed(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    self.layer_name = layer.name
                    break
        
        grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )
        return grad_model
    
    def compute_heatmap(self, image):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            loss = predictions[:, self.class_idx]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()

def preprocess_image(img_path, img_size=50):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array, img

def overlay_heatmap(heatmap, original_img, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (original_img.size[0], original_img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    
    original_img_array = np.array(original_img)
    overlay = cv2.addWeighted(original_img_array, 1-alpha, heatmap, alpha, 0)
    
    return overlay, heatmap

def generate_gradcam_visualization(model_path, img_path, class_names, img_size=50):
    model = load_model(model_path)
    
    img_array, original_img = preprocess_image(img_path, img_size)
    
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    gradcam = GradCAM(model, predicted_class_idx)
    heatmap = gradcam.compute_heatmap(img_array)
    
    overlay, heatmap_colored = overlay_heatmap(heatmap, original_img)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap_colored)
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay\n{class_names[predicted_class_idx]}\nConfidence: {confidence:.2%}')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'prediction': class_names[predicted_class_idx],
        'confidence': confidence,
        'heatmap': heatmap,
        'overlay': overlay
    }
