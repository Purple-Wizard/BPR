import argparse
from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    args = parse_arguments()

    compressed_images = fetch_from_local(args.path_to_compressed)

    try:
        decoder = load_model(args.decoder_model)
    except:
        print('Please train or set the path to an appropriate model')
        return
    
    decompressed_images = decoder.predict(compressed_images)

    if (args.save_local):
        save_images(decompressed_images, args.images_folder_name)

    if (args.console_out):
        show_images(decompressed_images, args.console_out)

def parse_arguments():
    parser = argparse.ArgumentParser(description='This is the decompression script for 128x128 images and local storage')
    parser.add_argument('--decoder_model', type=str, help='Path to decoder model', default='models/decoder.h5')
    parser.add_argument('--path_to_compressed', type=str, help='Path to compressed data (.npy)', default='test/compressed/test.npy')
    parser.add_argument('--console_out', type=int, help='Number of samples to show after the decompression', default=10)
    parser.add_argument('--save_local', type=bool, help='If the decompressed images should be saved', default=False)
    parser.add_argument('--images_folder_name', type=str, help='Name of the output folder', default='sample')

    return parser.parse_args()

def fetch_from_local(path):
    return np.load(path)

def save_images(images, folder_name):
    os.makedirs(f'decompressed/{folder_name}', exist_ok=True)

    for i, image in enumerate(images):
        images = (image.reshape(128, 128, 3) * 255).astype(np.uint8)
        images = Image.fromarray(images)
        images.save(f'decompressed/{folder_name}/re_img_{i}.png')

def show_images(images, n):
    n = min(n, len(images))
    cols = 3
    rows = n // cols
    rows += n % cols

    fig = plt.figure(figsize=(20, 20))

    for i in range(n):
        img = images[i]
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(img)
        ax.axis('off')

    plt.show()


if __name__ == '__main__':
    main()
