from db_connection import WeaviateConnection

import argparse
from keras.models import load_model

def main():
    args = parse_arguments()

    if (args.db):
        compressed_images = fetch_from_db(args.images_id, args.db)

    elif (args.local):
        compressed_images = fetch_from_local(args.path_to_compressed)

    try:
        decoder = load_model(args.decoder_model)
    except:
        print('Please train or set the path to an appropriate model')
        return
    
    decompressed_images = decoder.predict(compressed_images)

def parse_arguments():
    parser = argparse.ArgumentParser(description='This is the compression script for 128x128 images and storage')
    parser.add_argument('--images_id', type=str, help='Path to dataset (.png)', default='sample1')
    parser.add_argument('--decoder_model', type=str, help='Path to dataset (.png)', default='models/decoder.h5')
    parser.add_argument('--db', type=str, help='Number of epochs to run', default='http://localhost:8080')
    parser.add_argument('--local', type=bool, help='Peek into the preprocessing before the training', default=False)
    parser.add_argument('--save_local', type=bool, help='Peek into the preprocessing before the training', default=False)
    parser.add_argument('--console_out', type=bool, help='Number of samples to show after the validation', default=True)
    parser.add_argument('--path_to_compressed', type=str, help='Saves the trained models as .h5 (encoder, decoder)', default=1)

    return parser.parse_args()

def save_locally(images):
    pass

def console_out(images):
    pass

def fetch_from_db(images_id, connection):
    client = WeaviateConnection(connection)
    class_schema = client.schema.get('Image')
    return client.get_images(images_id)

def fetch_from_local(path):
    pass

def show_images(images, n):
    pass


if __name__ == '__main__':
    main()