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

#modelling VGG-16

