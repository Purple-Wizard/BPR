from db_connection import WeaviateConnection
from hashing_layer import HashingLayer
from utility import load_dataset

import argparse
from keras.models import load_model
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def main():
    args = parse_arguments()

    images = load_dataset(args.data_to_compress, args.n_images, compression=True)

    try:
        encoder = load_model(args.encoder_model, custom_objects={'HashingLayer': HashingLayer})
    except:
        print('Please train or set the path to an appropriate model')
        return

    compressed_img = encoder.predict(images)

    if (args.db):
        save_to_db(compressed_img, args.name_images, args.db)

def parse_arguments():
    parser = argparse.ArgumentParser(description='This is the compression script for 128x128 images and storage')
    parser.add_argument('--data_to_compress', type=str, help='Path to dataset (.png)', default='dataset/set2')
    parser.add_argument('--name_images', type=str, help='Path to dataset (.png)', default='sample')
    parser.add_argument('--encoder_model', type=str, help='Size of the dataset to use (25.000 pcs recommended)', default='models/encoder.h5')
    parser.add_argument('--db', type=str, help='Number of epochs to run', default='http://localhost:8080')
    parser.add_argument('--save_local', type=bool, help='Peek into the preprocessing before the training', default=False)
    parser.add_argument('--console_out', type=bool, help='Number of samples to show after the validation', default=True)
    parser.add_argument('--n_images', type=int, help='Saves the trained models as .h5 (encoder, decoder)', default=1)

    return parser.parse_args()

def save_to_db(images, name, connection):
    client = WeaviateConnection(connection)

    client.add_image(name, images)

def console_out():
    pass

def save_locally():
    pass


if __name__ == '__main__':
    main()