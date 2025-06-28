import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 1. Load the saved model
model = load_model("mnist_ann_model_2.h5")
print(" Model loaded.")

# 2. Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    
    img = np.array(img)
    img = 255 - img  # Optional: Invert if needed
    img = img / 255.0

    # Show image
    plt.imshow(img, cmap='gray')
    plt.title("Input Image")
    plt.axis('off')
    plt.show()

    # Reshape to (1, 28, 28, 1) if model expects channel dim
    if len(model.input_shape) == 4:
        img = img.reshape(1, 28, 28, 1)
    else:
        img = img.reshape(1, 28, 28)
    
    return img

# 3. Predict digit
def predict_digit(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    print(f" Predicted digit: {digit} (Confidence: {confidence:.2f}%)")

# 4. Usage
predict_digit('test0.jpg')
