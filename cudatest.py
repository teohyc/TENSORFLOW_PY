import tensorflow as tf
from tensorflow.python.client import device_lib

print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Built with GPU support:", tf.test.is_built_with_gpu_support())
print("GPUs detected:", tf.config.list_physical_devices('GPU'))
print("Local devices:\n", device_lib.list_local_devices())
