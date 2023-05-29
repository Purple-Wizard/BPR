import os
from keras.models import load_model
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from utility import load_dataset
from skimage.metrics import structural_similarity
from autoencoder import HashingLayer


NUM_OF_IMGS = 100
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
IMG_DIRECTORY = 'set2'
ENCODER_PATH = 'models/encoder.h5'
DECODER_PATH = 'models/decoder.h5'


def load_and_predict_images(image_directory, num_of_imgs, img_size, encoder_path, decoder_path):
    # Load the images
    data = load_dataset(image_directory, num_of_imgs, img_size)
    
    processed_data = data['processed']
    original_data = data['preprocessed']
    processed_data_arr = processed_data.batch(BATCH_SIZE)
    
    # Load the model
    encoder = load_model(encoder_path, custom_objects={'HashingLayer': HashingLayer})
    decoder = load_model(decoder_path)

    # List to store the original and predicted images    
    z = encoder.predict(processed_data_arr)
    predicted = decoder.predict(z)
        
    return original_data, predicted


def main():
    original, predicted = load_and_predict_images(IMG_DIRECTORY, NUM_OF_IMGS, IMG_SIZE, ENCODER_PATH, DECODER_PATH)
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    
    for i in range(1, 6):
        # compute the column index for the subplot
        col = i % 5
                
        # Original images in the first row
        axs[0, col].imshow(original[i])
        axs[0, col].axis('off')

        # Predicted images in the second row
        axs[1, col].imshow(predicted[i])
        axs[1, col].axis('off')

    plt.show()


if __name__ == '__main__':
  main()