import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random


def load_dataset(path: str, num_images: int, img_size: tuple = (128, 128), peek: bool = False, compression: bool = False):
    image_files = []

    for root, _, files in os.walk(path):
        image_files.extend([os.path.join(root, file) for file in files if file.lower().endswith('.png')])

    image_files = image_files[:num_images]
    
    original_images = [
        np.array(Image.open(file))
        for file in tqdm(image_files, desc='Loading original images', unit='images')
    ]

    resized = resize_images(original_images, img_size)

    if (compression):
        return tf.data.Dataset.from_tensor_slices((np.array(resized), np.array(resized))).batch(32)

    processed_images = preprocess_images(resized, peek)

    original_train, original_val, original_test = split_data(resized)
    processed_train, processed_val, processed_test = split_data(processed_images)

    train_dataset = tf.data.Dataset.from_tensor_slices((np.array(processed_train), np.array(original_train)))
    val_dataset = tf.data.Dataset.from_tensor_slices((np.array(processed_val), np.array(original_val)))
    test_dataset = tf.data.Dataset.from_tensor_slices((np.array(processed_test), np.array(original_test)))
    processed_dataset = tf.data.Dataset.from_tensor_slices((np.array(processed_images), np.array(processed_images)))

    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset,
        'original': resized,
        'processed': processed_dataset
    }

def resize_images(images, img_size):
    resized_imgs = []

    for img in tqdm(images, desc=f'Resizing images to {img_size[0]}x{img_size[1]}', unit='images'):
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        resized_imgs.append(img)

    resized_imgs = np.array(resized_imgs)
    resized_imgs = resized_imgs.astype('float32') / 255.0

    return resized_imgs

def display_images(images_lists, titles=['Original']):
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    for i in range(len(images_lists)):
        if (i > 3):
            break

        for j in range(len(images_lists[0])):
            img = images_lists[i][j]
            axs[i, j].imshow(img, cmap='jet')
            axs[i, j].set_title(titles[j])
    plt.tight_layout()
    plt.show()

def apply_salt_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    height, width, channel = image.shape

    num_salt = int(height * width * salt_prob)
    salt_coords = [random.randint(0, height - 1) for _ in range(num_salt)]
    salt_indices = np.unravel_index(salt_coords, (height, width))
    noisy_image[salt_indices] = 1.0

    num_pepper = int(height * width * pepper_prob)
    pepper_coords = [random.randint(0, height - 1) for _ in range(num_pepper)]
    pepper_indices = np.unravel_index(pepper_coords, (height, width))
    noisy_image[pepper_indices] = 0.0

    return noisy_image

def preprocess_images(images, pre_peek: bool):
    processed_images = []
    display_images_list = []
    titles = ['Original', 'S&P', 'Gaussian']

    for i, img in enumerate(tqdm(images, desc='Preprocessing images', unit='images')):
        salt_pepper = apply_salt_pepper_noise(img.copy(), 0.01, 0.01)
        
        blur_vis = cv2.GaussianBlur(salt_pepper, (3, 3), 0)

        display_images_list.append([img, salt_pepper, blur_vis])
        processed_images.append(blur_vis)

    if (pre_peek):
        display_images(display_images_list, titles)

    return np.array(processed_images, dtype=np.float32)

def split_data(images):
    print('Splitting data to train, val, and test...\n_________________________________________________________________')
    images = np.array(images)
    x_train, x_temp = train_test_split(images, test_size=0.4, random_state=42)
    x_val, x_test = train_test_split(x_temp, test_size=0.7, random_state=42)
    return x_train, x_val, x_test
