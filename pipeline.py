from preprocess import load_images
from load_for_bt import load_and_predict_images
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
from preprocess import resize_images
import tensorflow as tf


# Constants
DATASET_PATH = 'set4'
ENCODER_PATH = 'models/encoder.h5'
DECODER_PATH = 'models/decoder.h5'
BARLOWTWINS_PATH = 'models/barlow_twins.h5'
NUM_OF_IMGS = 10
ENCODER_IMG_SIZE = (128, 128)
BARLOWTWINS_IMG_SIZE = (32, 32)
BARLOWTWINS_BATCH_SIZE = 32
AUTO = tf.data.AUTOTUNE


# Load images for encoder
print('\nLoad and Preprocess... \n____________________________________________________________')
test_images = load_images(DATASET_PATH, NUM_OF_IMGS, ENCODER_IMG_SIZE)


# Predict with Autoencoder
print('\nPredict with Autoencoder... \n____________________________________________________________')
original, predicted = load_and_predict_images(DATASET_PATH, NUM_OF_IMGS, ENCODER_IMG_SIZE, ENCODER_PATH, DECODER_PATH)


# Visualize original and predicted images
print('\nVisualize original and predicted images... \n____________________________________________________________')
fig, axs = plt.subplots(2, 10, figsize=(40, 8))
for i in range(1, 11):
    col = i % 10
    
    # Original images in the first row
    axs[0, col].imshow(original[i])
    axs[0, col].axis('off')
    
    # Predicted images in the second row
    axs[1, col].imshow(predicted[i])
    axs[1, col].axis('off')
plt.show()


# Load BarlowTwins model
print('\nLoad BarlowTwins model... \n____________________________________________________________')
barlow_twins = load_model(BARLOWTWINS_PATH)


# Prepare image for BarlowTwins
dataset_one = resize_images(original, BARLOWTWINS_IMG_SIZE)
dataset_two = resize_images(predicted, BARLOWTWINS_IMG_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((dataset_one, dataset_two))
test_ds = test_ds.batch(32).prefetch(AUTO)


print('\nVisualize cross corelation... \n____________________________________________________________')
# Get feature vectors from the trained model
feature_vectors = barlow_twins.predict(test_ds)

# Calculate the cross-correlation matrix
cross_correlation = np.corrcoef(feature_vectors.T)

# Visualize the cross-correlation matrix
plt.figure(figsize=(10, 10))
plt.imshow(cross_correlation, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.title("Cross-correlation Matrix")
plt.xlabel("Feature Index")
plt.ylabel("Feature Index")
plt.show()