import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 1. Load the saved model
model = load_model("mnist_ann_model.h5")
print("✅ Model loaded.")

# 2. Function to preprocess the image
def preprocess_image(image_path):
    # Open and convert to grayscale ('L' mode)
    img = Image.open(image_path).convert('L')
    
    # Resize to 28x28 if needed
    img = img.resize((28, 28))

    # Invert pixel values (make digit white on black, if necessary)
    img = np.array(img)
    img = 255 - img  # Invert colors if digit is dark on light

    # Normalize to 0–1 range
    img = img / 255.0

    # Show the image
    plt.imshow(img, cmap='gray')
    plt.title("Input Image")
    plt.axis('off')
    plt.show()

    # Reshape to (1, 28, 28) as expected by model
    return img.reshape(1, 28, 28)

# 3. Predict digit
def predict_digit(image_path):
    preprocessed = preprocess_image(image_path)
    prediction = model.predict(preprocessed)
    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    print(f" Predicted digit: {digit} (Confidence: {confidence:.2f}%)")

# 4. Example usage
# Replace 'your_image.png' with your actual file path
predict_digit('test0.jpg')
