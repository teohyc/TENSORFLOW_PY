import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import layers

import tensorflow as tf
import matplotlib.pyplot as plt

SEED_VALUE = 42

#fix seed to make training deterministic
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

#load boston housing dataset
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

print(X_train.shape)
print("\n")
print("input features: ", X_train[0])
print("\n")
print("output target: ", y_train[0])

#use only one feature
boston_features = { "Average Number of Rooms": 5,}

X_train_1d = X_train[:, boston_features["Average Number of Rooms"]]
print(X_train_1d.shape)

X_test_1d = X_test[:, boston_features["Average Number of Rooms"]]
print(X_test_1d.shape)

# median price vs no. of rooms
plt.figure(figsize=(15,5))

plt.xlabel("average number of rooms")
plt.ylabel("median price [$K]")
plt.grid("on")
plt.scatter(X_train_1d[:], y_train, color="green", alpha=0.5)
plt.show()

#initiate model
model=Sequential()

#define the model consisting of a single neuron
model.add(Dense(units=1, input_shape=(1,)))

#display a summary of the model
model.summary()

#compile the model
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.005), loss='mse') 

#train the model
history= model.fit(
    X_train_1d,
    y_train,
    batch_size=16,
    epochs=101,
    validation_split=0.3,
)

#plotting out the loss
def plot_loss(history):
    plt.figure(figsize=(20,5))
    plt.plot(history.history['loss'], 'g', label='training loss')
    plt.plot(history.history['val_loss'], 'b', label='validation loss')
    plt.xlim([0, 100])
    plt.ylim([0, 300])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss(history)

#predict the median price of a home with [3, 4, 5, 6, 7] rooms
x = np.array([3, 4, 5, 6, 7])
y_pred = model.predict(x)
for idx in range( len(x) ):
    predicted_price = y_pred[idx].item()
    print(f"predicted price of a home with {x[idx]} rooms: ${int(predicted_price * 10 ) / 10 }K")

#generate feature data that spans the range of interest for independent variables
x = np.linspace(3, 9, 10)

#use the model to predict dependent variable
y = model.predict(x)

#plotting the predicted values
def plot_data(x_data, y_data, x, y , title=None):
    plt.figure(figsize=(15,5))
    plt.scatter(x_data, y_data, label='ground truth', color='green', alpha=0.5)
    plt.plot(x, y, color='k', label='model predictions')
    plt.xlim([3,9])
    plt.ylim([0,60])
    plt.xlabel('average number of rooms')
    plt.ylabel('price [$K]')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

plot_data(X_train_1d, y_train, x, y, title='training datasets')

#use the test data
plot_data(X_test_1d, y_test, x, y, title='test datasets')