import os
import pathlib
import random
import numpy as np
import matplotlib.pyplot as plt

import zipfile
import requests
import glob as glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow. keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.utils import image_dataset_from_directory

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from dataclasses import dataclass

from zipfile import ZipFile
from urllib.request import urlretrieve

SEED_VALUE = 41

#setting seed value to make training seterministic
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

#download and extract dataset
def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assets....", end="")

    #download zip file
    urlretrieve(url, save_path)

    try:
        #extract
        with ZipFile(save_path) as z:
            #extract zipfile in same directory
            z.extractall(os.path.split(save_path)[0])

        print("Done")

    except Exception as e:
        print("\nInvalid file.", e)

URL = r"https://www.dropbox.com/s/uzgh5g2bnz40o13/dataset_traffic_signs_40_samples_per_class.zip?dl=1"

dataset_path = os.path.join(os.getcwd(), "dataset_traffic_signs_40_samples_per_class")
asset_zip_path = os.path.join(os.getcwd(), "dataset_traffic_signs_40_samples_per_class.zip")

#download if asset ZIP doesn't exists
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)
else:
    print("Download already...")

#data class for data configuration

@dataclass(frozen=True)
class DatasetConfig:
    NUM_CLASSES: int = 43
    IMG_HEIGHT: int = 224
    IMG_WIDTH: int = 224
    CHANNELS: int = 3

    DATA_ROOT_TRAIN: str = os.path.join(dataset_path, "Train")
    DATA_ROOT_VALID: str = os.path.join(dataset_path, "Valid")
    DATA_ROOT_TEST: str = os.path.join(dataset_path, "Test")
    DATA_TEST_GT: str = os.path.join(dataset_path, "Test.csv")

@dataclass(frozen=True)
class TrainingConfig:
    BATCH_SIZE: int = 32
    EPOCHS: int = 101
    LEARNING_RATE: int = 0.0001
    DROPOUT: float = 0.6
    LAYERS_FINE_TUNE: int = 8

#create train and validation datasets

train_dataset = image_dataset_from_directory(directory=DatasetConfig.DATA_ROOT_TRAIN,
                                             batch_size=TrainingConfig.BATCH_SIZE,
                                             shuffle=True,
                                             seed=SEED_VALUE,
                                             label_mode='int', #use integer encoding
                                             image_size=(DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH)
                                             )

valid_dataset = image_dataset_from_directory(directory=DatasetConfig.DATA_ROOT_VALID,
                                             batch_size=TrainingConfig.BATCH_SIZE,
                                             shuffle=True,
                                             seed=SEED_VALUE,
                                             label_mode='int',
                                             image_size=(DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH)
                                             )
#display the class names from training dataset
print(train_dataset.class_names)

#display sample image from dataset
class_names = train_dataset.class_names
plt.figure(figsize=(18, 10))

#assume dataset batch_size is at least 32
num_rows = 4
num_cols = 8

#use take method to retrieve just first batch of data from the training portion of the dataset
for image_batch, labels_batch in train_dataset.take(1):
    #plot each images in the batch and the associated ground truth labels
    for i in range(num_rows * num_cols):
        ax = plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        truth_idx = labels_batch[i].numpy()
        plt.title(class_names[truth_idx])
        plt.axis("off")
plt.show()

#create test dataset
#load test ground truth labels

import pandas as pd

input_file = DatasetConfig.DATA_TEST_GT

dataset = pd.read_csv(input_file)
df = pd.DataFrame(dataset)
cols = [6]
df = df[df.columns[cols]]
ground_truth_ids = df["ClassId"].values.tolist()
print("Total number of Test labels: ", len(ground_truth_ids))
print(ground_truth_ids[0:10])

#map ground truth class id to id in train/valid datasets
#convert train/valid class names to int
class_names_int = list(map(int, train_dataset.class_names))

#create dictionary mapping ground truth ID to class name ID
gtid_2_cnidx = dict(zip(class_names_int, range(0, DatasetConfig.NUM_CLASSES)))

print(gtid_2_cnidx.items())

#convert ground truth id to id that maps correctly to same class
#in train/valid datasets
label_ids = []
for idx in range(len(ground_truth_ids)):
    label_ids.append(gtid_2_cnidx[ground_truth_ids[idx]])

print("original grouth truth class id: ", ground_truth_ids[0:10])
print("new mapping: ", label_ids[0:10])
print("")
print("train/valid dataset class names: ", train_dataset.class_names)

#create file path to test images
#get all path names to test images
image_paths = sorted(glob.glob(DatasetConfig.DATA_ROOT_TEST + os.sep + "*.png"))

print(len(image_paths))
print("")
#print first 5 image paths
for idx in range(5):
    print(image_paths[idx])

#combine images and labels to create test dataset
test_dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_ids))

#load and process images
def preprocess_image(image):
    #decode and resize image
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH])
    return image

def load_and_preprocess_image(path):
    #read image into memory as byte string
    image = tf.io.read_file(path)
    return preprocess_image(image)

def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label

#apply the functions above to the dataset
test_dataset = test_dataset.map(load_and_preprocess_from_path_label)

#set the batch size for dataset
test_dataset = test_dataset.batch(TrainingConfig.BATCH_SIZE)

#display image from test dataset
plt.figure(figsize=(18, 10))

#assume dataset batch_size is at least 32
num_rows = 4
num_cols = 8

#use take() to retrieve first batch
for image_batch, labels_batch in test_dataset.take(1):

    #plot each image and ground truth
    for i in range(num_rows * num_cols):
        ax = plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        truth_idx = labels_batch[i].numpy()
        plt.title(class_names[truth_idx])
        plt.axis("off")
plt.show()

#specify model input shape
input_shape = (DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH, DatasetConfig.CHANNELS)

print('Loading model with ImageNet weights...')
vgg16_conv_base = tf.keras.applications.vgg16.VGG16(input_shape=input_shape,
                                                    include_top=False, #we supply our own top 
                                                    weights='imagenet',
                                                    )
vgg16_conv_base.summary()

#freeze initial layers in convolutional base
#set all layers in convolutional base to Trainable
vgg16_conv_base.trainable = True

#specify number of layers to fine tune at the end of convolutional base
num_layers_fine_tune = TrainingConfig.LAYERS_FINE_TUNE
num_layers = len(vgg16_conv_base.layers)

#freeze the initail layers in convolutional base
for model_layer in vgg16_conv_base.layers[: num_layers - num_layers_fine_tune]:
    print(f"FREEZING LAYER: {model_layer}")
    model_layer.trainable = False

print("\n")
print(f"Configured to fine tune the last {num_layers_fine_tune} convolutional layers")
print("\n")

vgg16_conv_base.summary()


### VGG-16 ARCHITECTURE ###

#add classifier
inputs = tf.keras.Input(shape=input_shape)

x = tf.keras.applications.vgg16.preprocess_input(inputs)

x = vgg16_conv_base(x)

#flatten output of convolutional base
x = layers.Flatten()(x)

#add classifier
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(TrainingConfig.DROPOUT)(x)

#output layer
outputs = layers.Dense(DatasetConfig.NUM_CLASSES, activation="softmax")(x)

#final model
model_vgg16_finetune = keras.Model(inputs, outputs)
model_vgg16_finetune.summary()

#use intiger encoded labels
model_vgg16_finetune.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=TrainingConfig.LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

#train
training_results = model_vgg16_finetune.fit(train_dataset,
                                             epochs=TrainingConfig.EPOCHS,
                                             validation_data=valid_dataset,
                                             )

#plot training results
def plot_results(metrics, ylabel=None, ylim=None, metric_name=None, color=None):
    fig, ax = plt.subplots(figsize=(15, 4))

    if not (isinstance(metric_name, list) or isinstance(metric_name, tuple)):
        metrics = [metrics,]
        metric_name = [metric_name,]

    for idx, metric in enumerate(metrics):
        ax.plot(metric, color=color[idx])
    
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.xlim([0, TrainingConfig.EPOCHS -1])
    plt.ylim(ylim)
    #tailor x-axis tick marks
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.grid(True)
    plt.legend(metric_name)
    plt.show()
    plt.close()

#retrieve training results
train_loss = training_results.history["loss"]
train_acc = training_results.history["accuracy"]
valid_loss = training_results.history["val_loss"]
valid_acc = training_results.history["val_accuracy"]

plot_results(
    [train_loss, valid_loss],
    ylabel="Loss",
    ylim=[0.0, 5.0],
    metric_name=["Trainig Loss", "Validation Loss"],
    color=["g", "b"],
)

plot_results(
    [train_acc, valid_acc],
    ylabel="Accuracy",
    ylim=[0.0, 1.0],
    metric_name=["Training Accuracy", "Validation Accuracy"],
    color=["g", "b"],
)

print(f"Model valid accuracy: {model_vgg16_finetune.evaluate(valid_dataset)[1]*100:.3f}")
print(f"Model test accuracy: {model_vgg16_finetune.evaluate(test_dataset)[1]*100:.3f}")

#display sample prediction
def display_predictions(dataset, model, class_names):

    plt.figure(figsize=(20, 20))
    num_rows = 8
    num_cols = 8
    jdx = 0

    #evaluate two batches.
    for image_batch, labels_batch in dataset.take(2):
        print(image_batch.shape)

        #prediction for the current batch
        predictions = model.predict(image_batch)

        #loop over all images
        for idx in range(len(labels_batch)):
            pred_idx = tf.argmax(predictions[idx]).numpy()
            truth_idx = labels_batch[idx].numpy

            #set title color
            if pred_idx == truth_idx:
                color = "g"
            else:
                color = "r"

            jdx += 1

            if jdx > num_rows * num_cols:
                #break from loops during max image
                break

            ax = plt.subplot(num_rows, num_cols, jdx)
            title = str(class_names[truth_idx]) + " : " + str(class_names[pred_idx])

            title_obj = plt.title(title)
            plt.setp(title_obj, color=color)
            plt.axis("off")
            plt.imshow(image_batch[idx].numpy().astype("uint8"))

    plt.show()
    return

display_predictions(valid_dataset, model_vgg16_finetune, class_names)
display_predictions(test_dataset, model_vgg16_finetune, class_names) 
            