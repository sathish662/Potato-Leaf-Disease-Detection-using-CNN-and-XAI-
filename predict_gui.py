import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the trained model
model = load_model('potato_leaf_disease_model.h5')

# Define image size for model input
IMG_SIZE = 50

# Define class labels (update according to your model's training)
class_labels = ['Healthy', 'Early_Blight', 'Late_Blight']  # Example for 3 classes

def predict_image():
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    
    if file_path:  # If an image was selected
        # Read and preprocess the image
        img = cv2.imread(file_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to match the model input size
        img = np.array(img)  # Convert image to numpy array
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img / 255.0  # Normalize the image to [0,1]

        # Predict the image class
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)

        # Display the prediction result
        result_label.config(text=f"Prediction: {class_labels[predicted_class[0]]}")
        result_label.pack()

# Create the main window
root = tk.Tk()
root.title("Potato Leaf Disease Prediction")

# Create and pack the prediction button
predict_button = tk.Button(root, text="Select Image and Predict", command=predict_image)
predict_button.pack(pady=20)

# Create and pack the result label
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack()

# Start the Tkinter main loop
root.mainloop()
