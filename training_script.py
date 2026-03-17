import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Set paths
TRAIN_DIR = r'C:\plantdisease\PLD_3_Classes_256\Training'
TEST_DIR = r'C:\plantdisease\PLD_3_Classes_256\Testing'
VALIDATION_DIR = r'C:\plantdisease\PLD_3_Classes_256\Validation'

IMG_SIZE = 50
BATCH_SIZE = 32
EPOCHS = 20

# Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Required for correct order
)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 output classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator)

# Predict
test_generator.reset()
y_true = test_generator.classes
y_probs = model.predict(test_generator, steps=len(test_generator), verbose=1)
y_pred = np.argmax(y_probs, axis=1)

# Class names
class_names = list(test_generator.class_indices.keys())

# --- EVALUATION METRICS ---

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_true, y_pred, display_labels=class_names, cmap='Blues'
)
plt.title('Confusion Matrix - Potato Disease Detection')
plt.show()

# Classification Report
report = classification_report(y_true, y_pred, target_names=class_names)
print("\n📊 Classification Report:\n")
print(report)

# Save the model
model.save("potato_leaf_disease_model.h5")
print("\n✅ Model trained, evaluated, and saved!")
