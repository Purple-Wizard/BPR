from hashing_layer import HashingLayer
from utility import load_dataset

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Add, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from skimage.metrics import structural_similarity
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
import warnings
warnings.filterwarnings('ignore')

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


def train():
    args = parse_arguments()

    DATASET_PATH = args.path_to_dataset
    DATASET_SIZE = args.dataset_size
    N_EPOCH = args.n_epoch

    data = load_dataset(DATASET_PATH, DATASET_SIZE, img_size=(128, 128), peek=args.show_preprocess)

    train_data_arr = data['train'].batch(32)
    test_data_arr = data['test'].batch(32)
    validation_data_arr = data['val'].batch(32)

    encoder = create_encoder()
    decoder = create_decoder(encoder)

    model = Model(encoder.input, decoder(encoder.output))

    lr_scheduler = ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.1,
                        patience=5,
                        verbose=1,
                        mode='min',
                        min_delta=0.001,
                        cooldown=3,
                        min_lr=1e-6
                    )
    early_stopping = EarlyStopping(
                        monitor="val_loss",
                        min_delta=0.0001,
                        patience=12,
                        verbose=1,
                        mode='min',
                        restore_best_weights=True
                    )

    encoder.summary()
    decoder.summary()

    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mae')
    history_comp = model.fit(
        train_data_arr,
        epochs=N_EPOCH,
        validation_split=0.2,
        batch_size=32,
        validation_data=(validation_data_arr),
        verbose=1,
        callbacks=[lr_scheduler, early_stopping]
    )

    if (args.save_models):
        encoder.save('models/encoder.h5')
        decoder.save('models/decoder.h5')

    encoded = encoder.predict(test_data_arr)
    decoded = decoder.predict(encoded)

    predict_and_visualize(
        number_of_samples=args.n_visualize_result,
        history=history_comp,
        test_data=test_data_arr,
        decoded=decoded,
        encoded=encoded
    )

def parse_arguments():
    parser = argparse.ArgumentParser(description='This is the training and validating script for Hashing Autoencoder')
    parser.add_argument('--path_to_dataset', type=str, help='Path to dataset (.png)', default='dataset/set1')
    parser.add_argument('--dataset_size', type=int, help='Size of the dataset to use (25.000 pcs recommended)', default=25000)
    parser.add_argument('--n_epoch', type=int, help='Number of epochs to run', default=100)
    parser.add_argument('--show_preprocess', type=bool, help='Peek into the preprocessing before the training', default=True)
    parser.add_argument('--n_visualize_result', type=int, help='Number of samples to show after the validation', default=10)
    parser.add_argument('--save_models', type=bool, help='Saves the trained models as .h5 (encoder, decoder)', default=True)

    return parser.parse_args()

def conv_block(x, filters, strides, padding='same'):
    x = Conv2D(filters, (3,3), strides=strides, padding=padding)(x)
    x = Activation('relu')(x)

    return x

def conv_transpose_block(x, filters, strides, padding='same'):
    x = Conv2DTranspose(filters, (3,3), strides=strides, padding=padding)(x)
    x = Activation('relu')(x)

    return x

def create_encoder(input_shape=(128, 128, 3)):
    input_img = Input(shape=input_shape)
    a1 = conv_block(input_img, 32, 1)
    a2 = conv_block(a1, 64, 2)
    a3 = conv_block(a2, 128, 1)
    a4 = conv_block(a3, 128, 1)
    skip_1 = Add()([a4, a3])
    a5 = conv_block(skip_1, 64, 1)
    a6 = conv_block(a5, 3, 1)
    a7 = HashingLayer(2)(Model(input_img, a6).output)
    
    return tf.keras.Model(input_img, a7)

def create_decoder(hashing_model):
    input_shape = hashing_model.output.shape[1:]
    decoder_input = Input(shape=input_shape)
    a8 = conv_transpose_block(decoder_input, 32, 1)
    a9 = conv_transpose_block(a8, 128, 2)
    a10 = conv_transpose_block(a9, 64, 1)
    a11 = conv_transpose_block(a10, 64, 1)
    skip_2 = Add()([a10, a11])
    a12 = conv_transpose_block(skip_2, 3, 1)

    return tf.keras.Model(decoder_input, a12)

def predict_and_visualize(number_of_samples: int, history, test_data, decoded, encoded):
    test_iter = next(iter(test_data))
    random_indices = random.sample(range(test_iter[0].shape[0]), number_of_samples)
    plt.figure(figsize=(21, 7))
    ax = plt.subplot(3, 1, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    for i, idx in enumerate(tqdm(random_indices, desc='Predicting and visualising results', unit='samples')):
        ax = plt.subplot(3, number_of_samples, i + number_of_samples + 1)
        test_sample = test_iter[0][idx]
        encoded_sample = encoded[idx]
        decoded_sample = decoded[idx]
        original_sample = test_iter[1][idx].numpy()

        plt.imshow(test_sample)
        ssims = [structural_similarity(
            original_sample.reshape(128, 128, 3),
            decoded_sample.reshape(128, 128, 3),
            data_range=1,
            multichannel=True,
            win_size=3
        ) for i in range(len(decoded))]

        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, number_of_samples, i + 1 + 2 * number_of_samples)
        plt.imshow(decoded_sample)
        plt.title(f'Avg SSIM: {np.mean(ssims):.4f}', fontsize=10)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        og_size = test_sample.numpy().size * test_sample.numpy().itemsize
        enc_size = encoded_sample.size * encoded_sample.itemsize
        dec_size = decoded_sample.size * decoded_sample.itemsize
        print(f'Original: {og_size} bytes | Encoded: {enc_size} bytes | Decoded: {dec_size} bytes')

    plt.show()


if __name__ == '__main__':
    train()
