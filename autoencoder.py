# Imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
import numpy as np
from tensorflow.keras.layers import Input, Add, Layer
from tensorflow.keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from skimage.metrics import structural_similarity
from preprocess import load_images
import random


# Constants

EPOCHS = 100
BATCH_SIZE = 50
IMG_SIZE = (128, 128)
NUM_OF_IMGS = 2000
DATASET_PATH = 'ae_dataset'


def main():
    data = load_images(DATASET_PATH, NUM_OF_IMGS, IMG_SIZE)
    
    train_data_arr = data['train']
    test_data_arr = data['test']
    validation_data_arr = data['val']
    
    train_data_arr = train_data_arr.batch(BATCH_SIZE)
    test_data_arr = test_data_arr.batch(BATCH_SIZE)
    validation_data_arr = validation_data_arr.batch(BATCH_SIZE)

    encoder, decoder, model_history = build_model(training_data = train_data_arr, epochs = EPOCHS, batch_size = BATCH_SIZE, validation_data = validation_data_arr)
    visualize_loss(model_history)
    test_and_visualize(encoder = encoder, decoder = decoder, test_data = test_data_arr)
    save_model(encoder = encoder, decoder = decoder)


def conv_block(x, filters, strides, padding='same'):
    x = tf.keras.layers.Conv2D(filters, (3,3), strides=strides, padding=padding)(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x


def conv_transpose_block(x, filters, strides, padding='same'):
    x = tf.keras.layers.Conv2DTranspose(filters, (3,3), strides=strides, padding=padding)(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x


class HashingLayer(Layer):
    def __init__(self, output_channels, **kwargs):
        super(HashingLayer, self).__init__(**kwargs)
        self.output_channels = output_channels

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(self.output_channels, (1, 1), strides=(1, 1), padding='same')
        super(HashingLayer, self).build(input_shape)

    def call(self, x):
        return self.conv(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.output_channels)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "output_channels": self.output_channels}


def create_encoder(input_shape=(128, 128, 3)):
    input_img = Input(shape=input_shape)
    a1 = conv_block(input_img, 32, 1)
    a3 = conv_block(a1, 64, 2)
    a5 = conv_block(a3, 128, 1)
    a7 = conv_block(a5, 128, 1)
    skip_0 = Add()([a7, a5])
    a9 = conv_block(skip_0, 64, 1)
    a11 = conv_block(a9, 3, 1)
    a12 = HashingLayer(2)(Model(input_img, a11).output)
    
    return tf.keras.Model(input_img, a12)


def create_decoder(hashing_model):
    input_shape = hashing_model.output.shape[1:]
    decoder_input = Input(shape=input_shape)
    a13 = conv_transpose_block(decoder_input, 32, 1)
    a15 = conv_transpose_block(a13, 128, 2)
    a17 = conv_transpose_block(a15, 64, 1)
    a19 = conv_transpose_block(a17, 64, 1)
    skip_1 = Add()([a17, a19])
    a21 = conv_transpose_block(skip_1, 3, 1)

    return tf.keras.Model(decoder_input, a21)


def build_model(training_data, epochs, batch_size,validation_data):
    encoder = create_encoder()
    decoder = create_decoder(encoder)

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
                        patience=10,
                        verbose=1,
                        mode='min',
                        restore_best_weights=True
        )

    model = Model(encoder.input, decoder(encoder.output))

    print("Encoder...\n_________________________________________________________________")
    print(encoder.summary())
    print("Decoder...\n_________________________________________________________________")
    print(decoder.summary())

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mae')
    history_comp = model.fit(
                        training_data, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=validation_data, 
                        verbose=1, 
                        callbacks=[lr_scheduler, early_stopping])

    return encoder, decoder, history_comp


def visualize_loss(model_history):
    plt.figure(figsize=(21, 7))
    plt.plot(model_history.history['loss'], label='Training Loss')
    plt.plot(model_history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()
    
    
def test_and_visualize(encoder, decoder, test_data):
    encoded = encoder.predict(test_data)
    decoded = decoder.predict(encoded)
    
    n = 10
    test_iter = next(iter(test_data))
    random_indices = random.sample(range(test_iter[0].shape[0]), n)
    plt.figure(figsize=(21, 7))
    
    for i, idx in enumerate(random_indices):
        ax = plt.subplot(3, n, i + n + 1)
        plt.imshow(test_iter[0][idx])
        ssims = [structural_similarity(test_iter[1][idx].numpy().reshape(128, 128, 3), decoded[idx].reshape(128, 128, 3), data_range=1, multichannel=True, win_size=3) for i in range(len(decoded))]
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(decoded[idx])
        plt.title(f'Avg SSIM: {np.mean(ssims):.4f}', fontsize=10)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        og_size = test_iter[0][idx].numpy().size * test_iter[0][idx].numpy().itemsize
        enc_size = encoded[idx].size * encoded[idx].itemsize
        dec_size = decoded[idx].size * decoded[idx].itemsize
        print(f'Original: {og_size} | Encoded: {enc_size} bytes | Decoded: {dec_size} bytes')

    plt.show()


def save_model(encoder, decoder):
    encoder.save('models/encoder.h5')
    decoder.save('models/decoder.h5')

if __name__ == '__main__':
  main()