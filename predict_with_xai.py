import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from xai_gradcam import generate_gradcam_visualization
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

model_path = 'potato_leaf_disease_model.h5'
model = load_model(model_path)
IMG_SIZE = 50
class_names = ['Early_Blight', 'Healthy', 'Late_Blight']

def predict_and_explain():
    file_path = filedialog.askopenfilename(
        title="Select Image", 
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    
    if not file_path:
        return
    
    try:
        result = generate_gradcam_visualization(model_path, file_path, class_names, IMG_SIZE)
        
        result_text = f"Prediction: {result['prediction']}\nConfidence: {result['confidence']:.2%}"
        result_label.config(text=result_text)
        
        img = Image.open('gradcam_result.png')
        img = img.resize((600, 200), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        
        canvas.delete("all")
        canvas.create_image(300, 100, image=photo)
        canvas.image = photo
        
        messagebox.showinfo("Success", "Grad-CAM visualization generated successfully!")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error processing image: {str(e)}")

root = tk.Tk()
root.title("Potato Disease Detection with XAI")
root.geometry("700x500")

predict_button = tk.Button(root, text="Select Image & Generate XAI", command=predict_and_explain, font=("Arial", 12))
predict_button.pack(pady=20)

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

canvas = tk.Canvas(root, width=600, height=200, bg='white')
canvas.pack(pady=20)

root.mainloop()
