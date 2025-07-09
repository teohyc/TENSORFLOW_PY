import random
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

SEED_VALUE = 42
# fix seed value
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

#load cifar10 dataset
(_, _), (X_test, y_test) = cifar10.load_data()

#normalizing images to [0, 1]
X_test = X_test.astype("float32") / 255

#convert labels to one-hot encoding
y_test = to_categorical(y_test)

reloaded_model = models.load_model('open_cv_class_3_model.h5')

#evaluating model
test_loss, test_acc = reloaded_model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc*100:.3f}")

def evaluate_model(dataset, model):
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    
    num_rows = 3
    num_cols = 6

    #retrieve no. of images from dataset
    data_batch = dataset[0 : num_rows * num_cols]

    #get predictions from model
    predictions = model.predict(data_batch)

    plt.figure(figsize=(20, 8))
    num_matches = 0

    for idx in range(num_rows * num_cols):
        ax = plt.subplot(num_rows, num_cols, idx + 1)
        plt.axis("off")
        plt.imshow(data_batch[idx])

        pred_idx = tf.argmax(predictions[idx]).numpy()
        truth_idx = np.nonzero(y_test[idx])

        title = str(class_names[truth_idx[0][0]]) + ":" + str(class_names[pred_idx])
        title_obj = plt.title(title, fontdict={"fontsize": 13})

        if pred_idx == truth_idx:
            num_matches +=1
            plt.setp(title_obj, color="g")
        else:
            plt.setp(title_obj, color="r")

        acc = num_matches / (idx +1)

    print("Prediction accuracy:", int(100 * acc) / 100)
    plt.show()

    return

evaluate_model(X_test, reloaded_model)

#generate prediction for the test dataset
predictions = reloaded_model.predict(X_test)

#for each sample image, select the class label with the highest probability
predicted_labels = [np.argmax(i) for i in predictions]

#convert one-hot encoding to integers
y_test_integer_labels = tf.argmax(y_test, axis=1)

#generate confusion matrix
cm =tf.math.confusion_matrix(labels=y_test_integer_labels, predictions=predicted_labels)

#plot the confusion matrix as heat map
plt.figure(figsize=[12, 6])

#use seaborn heatmap
import seaborn as sn

sn.heatmap(cm, annot=True, fmt="d", annot_kws={"size": 12})
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()