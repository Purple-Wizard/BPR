import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras import Model, Input
from keras.layers import (
    Conv2D,
    Dense,
    BatchNormalization,
    ReLU,
    Add,
    Activation,
    GlobalAvgPool2D,
)
from keras.regularizers import l2
from load_for_bt import load_and_predict_images
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from PIL import Image
from utility import resize_images


## HYPERPARAMETER AND CONSTANTS

NUM_OF_IMGS = 17000
ENCODER_INPUT_SIZE = (128, 128)
BARLOW_TWINS_INPUT_SIZE = (64, 64)
INPUT_SHAPE = (64, 64, 3)
IMG_DIRECTORY = 'dataset/set3'
ENCODER_PATH = 'models/encoder.h5'
DECODER_PATH = 'models/decoder.h5'
PROJECT_DIM = 2048
BATCH_SIZE = 200
EPOCHS = 500
AUTO = tf.data.AUTOTUNE
CROP_TO = 32
SEED = 42
WEIGHT_DECAY = 5e-4


def main():
    ssl_ds, X_train, test_ds = data_loader(image_directory = IMG_DIRECTORY, num_of_imgs = NUM_OF_IMGS, encoder_img_size = ENCODER_INPUT_SIZE, encoder_path = ENCODER_PATH, decoder_path = DECODER_PATH, barlow_twims_img_size = BARLOW_TWINS_INPUT_SIZE)
    barlow_twins, history = build_model(ssl_ds, X_train, INPUT_SHAPE)
    visulaize_loss(history)
    save_model(barlow_twins)
    test_and_visualize(test_ds, barlow_twins)

## DATA LOADER
def data_loader(image_directory, num_of_imgs, encoder_img_size, encoder_path, decoder_path, barlow_twims_img_size):
    original, predicted = load_and_predict_images(image_directory, num_of_imgs, encoder_img_size, encoder_path, decoder_path)

    original = np.squeeze(original)
    predicted = np.squeeze(predicted)
    
    dataset_one = resize_images(original, barlow_twims_img_size)
    dataset_two = resize_images(predicted, barlow_twims_img_size)
    
    # Ratio of samples for training
    train_ratio = 0.8
    
    # Determine the number of samples for training
    num_train_samples = int(num_of_imgs * train_ratio)
    
    # Split the first dataset
    X_train = dataset_one[:num_train_samples]
    X_test = dataset_one[num_train_samples:]

    # Split the second dataset
    y_train = dataset_two[:num_train_samples]
    y_test = dataset_two[num_train_samples:]
    
    print('Train and test samples for the first network: ', len(X_train), len(X_test))
    print('Train and test samples for the second network: ', len(y_train), len(y_test))
    
    ssl_ds_one = tf.data.Dataset.from_tensor_slices(X_train)
    ssl_ds_one = (ssl_ds_one.shuffle(1024, seed=SEED).batch(BATCH_SIZE).prefetch(AUTO))

    ssl_ds_two = tf.data.Dataset.from_tensor_slices(y_train)
    ssl_ds_two = (ssl_ds_two.shuffle(1024, seed=SEED).batch(BATCH_SIZE).prefetch(AUTO))

    # We then zip both of these datasets.
    ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))
    
    test_one = resize_images(original, barlow_twims_img_size)
    test_two = resize_images(predicted, barlow_twims_img_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_one, test_two))
    test_ds = test_ds.batch(32).prefetch(AUTO)
        
    return ssl_ds, X_train, test_ds


# Loss Function

def off_diagonal(x):
    n = tf.shape(x)[0]
    flattened = tf.reshape(x, [-1])[:-1]
    off_diagonals = tf.reshape(flattened, (n-1, n+1))[:, 1:]
    return tf.reshape(off_diagonals, [-1])


def normalize_repr(z):
    z_norm = (z - tf.reduce_mean(z, axis=0)) / tf.math.reduce_std(z, axis=0)
    return z_norm


def compute_loss(z_a, z_b, lambd):
    # Get batch size and representation dimension.
    batch_size = tf.cast(tf.shape(z_a)[0], z_a.dtype)
    repr_dim = tf.shape(z_a)[1]

    # Normalize the representations along the batch dimension.
    z_a_norm = normalize_repr(z_a)
    z_b_norm = normalize_repr(z_b)

    # Cross-correlation matrix.
    c = tf.matmul(z_a_norm, z_b_norm, transpose_a=True) / batch_size

    # Loss.
    on_diag = tf.linalg.diag_part(c) + (-1)
    on_diag = tf.reduce_sum(tf.pow(on_diag, 2))
    off_diag = off_diagonal(c)
    off_diag = tf.reduce_sum(tf.pow(off_diag, 2))
    loss = on_diag + (lambd * off_diag)
    return loss 


# Build the model

class BarlowTwins(tf.keras.Model):
    def __init__(self, encoder, lambd=5e-3):
        super(BarlowTwins, self).__init__()
        self.encoder = encoder
        self.lambd = lambd
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        # Unpack the data.
        ds_one, ds_two = data

        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z_a, z_b = self.encoder(ds_one, training=True), self.encoder(ds_two, training=True)
            loss = compute_loss(z_a, z_b, self.lambd) 

        # Compute gradients and update the parameters.
        gradients = tape.gradient(loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


# Set up parameters

class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Implements an LR scheduler that warms up the learning rate for some training steps
    (usually at the beginning of the training) and then decays it
    with CosineDecay (see https://arxiv.org/abs/1608.03983)
    """

    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        learning_rate = (
            0.5
            * self.learning_rate_base
            * (
                1
                + tf.cos(
                    self.pi
                    * (tf.cast(step, tf.float32) - self.warmup_steps)
                    / float(self.total_steps - self.warmup_steps)
                )
            )
        )

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )


def stem(inputs):
    """Construct Stem Convolutional Group
    inputs : the input vector
    """
    x = Conv2D(
        16,
        (3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(WEIGHT_DECAY),
    )(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def learner(x, n_blocks):
    """Construct the Learner
    x          : input to the learner
    n_blocks   : number of blocks in a group
    """
    # First Residual Block Group of 16 filters (Stage 1)
    # Quadruple (4X) the size of filters to fit the next Residual Group
    x = residual_group(x, 16, n_blocks, strides=(1, 1), n=4)

    # Second Residual Block Group of 64 filters (Stage 2)
    # Double (2X) the size of filters and reduce feature maps by 75% (strides=2) to fit the next Residual Group
    x = residual_group(x, 64, n_blocks, n=2)

    # Third Residual Block Group of 64 filters (Stage 3)
    # Double (2X) the size of filters and reduce feature maps by 75% (strides=2) to fit the next Residual Group
    x = residual_group(x, 128, n_blocks, n=2)
    return x


def residual_group(x, n_filters, n_blocks, strides=(2, 2), n=2):
    """Construct a Residual Group
    x         : input into the group
    n_filters : number of filters for the group
    n_blocks  : number of residual blocks with identity link
    strides   : whether the projection block is a strided convolution
    n         : multiplier for the number of filters out
    """
    # Double the size of filters to fit the first Residual Group
    x = projection_block(x, n_filters, strides=strides, n=n)

    # Identity residual blocks
    for _ in range(n_blocks):
        x = identity_block(x, n_filters, n)
    return x


def identity_block(x, n_filters, n=2):
    """Construct a Bottleneck Residual Block of Convolutions
    x        : input into the block
    n_filters: number of filters
    n        : multiplier for filters out
    """
    # Save input vector (feature maps) for the identity link
    shortcut = x

    ## Construct the 1x1, 3x3, 1x1 residual block (fig 3c)

    # Dimensionality reduction
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        n_filters,
        (1, 1),
        strides=(1, 1),
        use_bias=False,
        kernel_regularizer=l2(WEIGHT_DECAY),
    )(x)

    # Bottleneck layer
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        n_filters,
        (3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(WEIGHT_DECAY),
    )(x)

    # Dimensionality restoration - increase the number of output filters by 2X or 4X
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        n_filters * n,
        (1, 1),
        strides=(1, 1),
        use_bias=False,
        kernel_regularizer=l2(WEIGHT_DECAY),
    )(x)

    # Add the identity link (input) to the output of the residual block
    x = Add()([x, shortcut])
    return x


def projection_block(x, n_filters, strides=(2, 2), n=2):
    """Construct a Bottleneck Residual Block with Projection Shortcut
    Increase the number of filters by 2X (or 4X on first stage)
    x        : input into the block
    n_filters: number of filters
    strides  : whether the first convolution is strided
    n        : multiplier for number of filters out
    """
    # Construct the projection shortcut
    # Increase filters by 2X (or 4X) to match shape when added to output of block
    shortcut = Conv2D(
        n_filters * n,
        (1, 1),
        strides=strides,
        use_bias=False,
        kernel_regularizer=l2(WEIGHT_DECAY),
    )(x)

    ## Construct the 1x1, 3x3, 1x1 convolution block

    # Dimensionality reduction
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        n_filters,
        (1, 1),
        strides=(1, 1),
        use_bias=False,
        kernel_regularizer=l2(WEIGHT_DECAY),
    )(x)

    # Bottleneck layer - feature pooling when strides=(2, 2)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        n_filters,
        (3, 3),
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(WEIGHT_DECAY),
    )(x)

    # Dimensionality restoration - increase the number of filters by 2X (or 4X)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        n_filters * n,
        (1, 1),
        strides=(1, 1),
        use_bias=False,
        kernel_regularizer=l2(WEIGHT_DECAY),
    )(x)

    # Add the projection shortcut to the output of the residual block
    x = Add()([shortcut, x])
    return x


def projection_head(x, hidden_dim=128):
    """Constructs the projection head."""
    for i in range(2):
        x = Dense(
            hidden_dim,
            name=f"projection_layer_{i}",
            kernel_regularizer=l2(WEIGHT_DECAY),
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    outputs = Dense(hidden_dim, name="projection_output")(x)
    return outputs


def prediction_head(x, hidden_dim=128, mx=4):
    """Constructs the prediction head."""
    x = BatchNormalization()(x)
    x = Dense(
        hidden_dim // mx,
        name=f"prediction_layer_0",
        kernel_regularizer=l2(WEIGHT_DECAY),
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dense(
        hidden_dim,
        name="prediction_output",
        kernel_regularizer=l2(WEIGHT_DECAY),
    )(x)
    return x


# -------------------
# Model      | n   |
# ResNet20   | 2   |
# ResNet56   | 6   |
# ResNet110  | 12  |
# ResNet164  | 18  |
# ResNet1001 | 111 |


def get_network(n=2, hidden_dim=128, use_pred=False, return_before_head=True, input_shape=INPUT_SHAPE):
    depth = n * 9 + 2
    n_blocks = ((depth - 2) // 9) - 1

    # The input tensor
    inputs = Input(shape=input_shape)
    x = Rescaling(scale=1.0 / 127.5, offset=-1)(inputs)

    # The Stem Convolution Group
    x = stem(x)

    # The learner
    x = learner(x, n_blocks)

    # Projections
    trunk_output = GlobalAvgPool2D()(x)
    projection_outputs = projection_head(trunk_output, hidden_dim=hidden_dim)

    if return_before_head:
        model = Model(inputs, [trunk_output, projection_outputs])
    else:
        model = Model(inputs, projection_outputs)

    # Predictions
    if use_pred:
        prediction_outputs = prediction_head(projection_outputs)
        if return_before_head:
            model = Model(inputs, [projection_outputs, prediction_outputs])
        else:
            model = Model(inputs, prediction_outputs)

    return model


def build_model(ssl_ds, X_train, input_shape):
    STEPS_PER_EPOCH = len(X_train) // BATCH_SIZE
    TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS
    WARMUP_EPOCHS = int(EPOCHS * 0.1)
    WARMUP_STEPS = int(WARMUP_EPOCHS * STEPS_PER_EPOCH)

    lr_decayed_fn = WarmUpCosine( 
        learning_rate_base=1e-3,
        total_steps=EPOCHS * STEPS_PER_EPOCH,
        warmup_learning_rate=0.0,
        warmup_steps=WARMUP_STEPS)

    # Visualize the LR schedule
    plt.plot(lr_decayed_fn(tf.range(EPOCHS*STEPS_PER_EPOCH, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.show()

    resnet_enc = get_network(hidden_dim=PROJECT_DIM, use_pred=False, return_before_head=False, input_shape=input_shape)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn, momentum=0.9)

    barlow_twins = BarlowTwins(resnet_enc)
    barlow_twins.compile(optimizer=optimizer)
    history = barlow_twins.fit(ssl_ds, epochs=EPOCHS)
    
    return barlow_twins.encoder, history


def visulaize_loss(model_history):
    plt.plot(model_history.history["loss"])
    plt.grid()
    plt.title("Barlow Twin Loss")
    plt.show()


def test_and_visualize(X_test, barlow_twins):
    # Get feature vectors from the trained model
    feature_vectors = barlow_twins.predict(X_test)

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


def save_model(barlow_twins):
    barlow_twins.save('models/barlow_twins.h5')


if __name__ == '__main__':
   main()