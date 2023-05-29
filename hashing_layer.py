from keras.layers import Conv2D, Layer
import tensorflow as tf
import warnings

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

class HashingLayer(Layer):
    def __init__(self, output_channels, **kwargs):
        super(HashingLayer, self).__init__(**kwargs)
        self.output_channels = output_channels

    def build(self, input_shape):
        self.conv = Conv2D(self.output_channels, (1, 1), strides=(1, 1), padding='same')
        super(HashingLayer, self).build(input_shape)

    def call(self, x):
        return self.conv(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.output_channels)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "output_channels": self.output_channels}
