from hashing_layer import HashingLayer
from utility import load_dataset

import argparse
from keras.models import load_model
import numpy as np
import os
import warnings
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

warnings.filterwarnings('ignore')


def main():
    args = parse_arguments()

    images = load_dataset(args.data_to_compress, compression=True)

    try:
        encoder = load_model(args.encoder_model, custom_objects={'HashingLayer': HashingLayer})
    except:
        print('Please train or set the path to an appropriate model')
        return

    compressed_img = encoder.predict(images)

    if (args.save_local):
        save_locally(compressed_img, args.name_images)

def parse_arguments():
    parser = argparse.ArgumentParser(description='This is the compression script for 128x128 images and storage')
    parser.add_argument('--data_to_compress', type=str, help='Path to images folder (.png)', default='test/gmaps_test')
    parser.add_argument('--name_images', type=str, help='Name of the compressed file (.npy)', default='sample')
    parser.add_argument('--encoder_model', type=str, help='Path to encoder model', default='models/encoder.h5')
    parser.add_argument('--save_local', type=bool, help='If the compression should be saved locally', default=True)

    return parser.parse_args()

def save_locally(images, name):
    os.makedirs('compressed', exist_ok=True)
    np.save(f'compressed/{name}.npy', images)

if __name__ == '__main__':
    main()