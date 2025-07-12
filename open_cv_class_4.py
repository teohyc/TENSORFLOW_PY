import tensorflow as tf
import matplotlib.pyplot as plt
import glob as glob
import os

from zipfile import ZipFile
from urllib.request import urlretrieve

#download assets
def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assets....", end="")
    
    #download zip file using urllib package
    urlretrieve(url, save_path)
    
    try:
        #extracting zip file using zipfile package
        with ZipFile(save_path) as z:
            
            #extract zip file contents in the same directory
            z.extractall(os.path.split(save_path)[0])
            
        print("Done")
        
    except Exception as e:
        print("\nInvalid file.", e)


#URL
URL = r"https://www.dropbox.com/s/8srx6xdjt9me3do/TF-Keras-Bootcamp-NB07-assets.zip?dl=1"

#determining path
asset_zip_path = os.path.join(os.getcwd(), "TF-Bootcamp-NB07-assets.zip")
print("Directory at: ", os.getcwd())
print("Data path at: ", asset_zip_path)

#download if asset zip does not exist
if not os.path.exists(asset_zip_path):
        download_and_unzip(URL, asset_zip_path)
else:
     print("Already downloaded")

#read sample images
#store all image paths in a list
image_paths = sorted(glob.glob("images" + os.sep + "*.png"))
print(image_paths)

#display sample images
plt.figure(figsize=(18, 6))
for idx, image_path in enumerate(image_paths):
     image = plt.imread(image_path)
     plt.subplot(2, 4, idx + 1)
     plt.imshow(image)
     plt.axis('off')
plt.show()

#Loading pre-trained models
model_vgg16 = tf.keras.applications.vgg16.VGG16()
model_resnet50 = tf.keras.applications.resnet50.ResNet50()
model_inception_v3 = tf.keras.applications.inception_v3.InceptionV3()

print(model_vgg16.input_shape)
print(model_resnet50.input_shape)
print(model_inception_v3.input_shape)

#batch processing
def process_images(model, image_paths, size, preprocess_input, display_top_k=False, top_k=2):
     
    plt.figure(figsize=(20,7))
    for idx, image_path in enumerate(image_paths):

        #READ image
        tf_image = tf.io.read_file(image_path)

        #DECODE the above 'tf_image' from a Bytes string to a numerical tensor
        decoded_image = tf.image.decode_image(tf_image)

        #RESIZE the image to the model requirement
        image_resized = tf.image.resize(decoded_image, size)

        #ADD BATCH dimension to first axis
        image_batch = tf.expand_dims(image_resized, axis=0)

        #PREPROCESS the input image
        image_batch = preprocess_input(image_batch)

        #forward pass through the model to make prediction
        preds = model.predict(image_batch)

        #decode (and rank the  top-k) predictions
        #return a list of tuples: (class ID, class description, probability)
        decoded_preds = tf.keras.applications.imagenet_utils.decode_predictions(
            preds=preds,
            top=5
        ) 

        if display_top_k == True:
            for jdx in range(top_k):
                 print("Top {} predicted class: Pr(Class={:20} [index={:4}]) = {:5.2f}".format(jdx + 1, decoded_preds[0][jdx][1], jdx, decoded_preds[0][jdx][2] * 100))

        plt.subplot(3, 4, idx + 1)
        plt.imshow(decoded_image)
        plt.axis('off')
        label = decoded_preds[0][0][1]
        score = decoded_preds[0][0][2]
        title = label + ' ' + str('{:.2f}%'.format(score))
        plt.title(title, fontsize=16)
    plt.show()

#vgg16
model = model_vgg16
size = (224, 224)
preprocess_input = tf.keras.applications.vgg16.preprocess_input
process_images(model, image_paths, size, preprocess_input)

#resnet50
model = model_resnet50
size = (224, 224)
preprocess_input = tf.keras.applications.resnet50.preprocess_input
process_images(model, image_paths, size, preprocess_input)

#inception-v3
model = model_inception_v3
size = (299, 299)
preprocess_input = tf.keras.applications.inception_v3.preprocess_input
process_images(model, image_paths, size, preprocess_input, display_top_k=True)
