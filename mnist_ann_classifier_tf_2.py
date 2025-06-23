import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize to [0, 1]
y_train, y_test = to_categorical(y_train), to_categorical(y_test)  # One-hot encoding

# 2. Improved ANN architecture
model = Sequential([
    Flatten(input_shape=(28, 28)),

    Dense(512, activation='relu'),
    BatchNormalization(),     # Helps stabilize and accelerate training
    Dropout(0.3),              # Prevents overfitting by randomly disabling neurons

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(10, activation='softmax')  # Output layer for 10 classes
])

# 3. Compile the model
model.compile(
    optimizer='adam',                    # Adaptive optimizer
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Train the model
history = model.fit(
    x_train, y_train,
    epochs=20,                           # More epochs = better convergence
    batch_size=128,
    validation_data=(x_test, y_test)
)

# 5. Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title("Loss over Epochs")
plt.grid(); plt.show()

# 6. Save model
model.save("mnist_ann_model_2.h5")
print("Saved improved TensorFlow model to mnist_ann_model_2.h5")
