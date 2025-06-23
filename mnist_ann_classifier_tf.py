import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. Load the MNIST dataset (handwritten digits 0–9)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalize the images to values between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Convert labels to one-hot encoding (e.g., 3 => [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 4. Build a simple ANN model
model = Sequential([
    Flatten(input_shape=(28, 28)),     # Flatten 28x28 images into 784-dim vectors
    Dense(128, activation='relu'),     # Hidden layer with 128 neurons
    Dense(64, activation='relu'),      # Another hidden layer
    Dense(10, activation='softmax')    # Output layer for 10 digit classes
])

# 5. Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Good for classification
    metrics=['accuracy']
)

# 6. Train the model and store the training history
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),  # Check performance on test set during training
    epochs=10,                         # You can increase this for better accuracy
    batch_size=32,
    verbose=1
)

# 7. Plot training & validation loss over epochs
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 8. Save the trained model as a .h5 file
model.save("mnist_ann_model.h5")
print("✅ Model saved to 'mnist_ann_model.h5'")