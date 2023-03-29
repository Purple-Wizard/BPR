import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten the images
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

# Apply PCA preprocessing
pca_components = 450
pca = PCA(n_components=pca_components)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# Build the binary autoencoder
input_dim = pca_components
encoding_dim = 64
threshold = 0.5

input_img = tf.keras.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
decoded = layers.Dense(input_dim, activation='relu')(encoded)

# Define the autoencoder and encoder models
autoencoder = tf.keras.Model(input_img, decoded)
encoder = tf.keras.Model(input_img, encoded)

# Compile the autoencoder model
autoencoder.compile(optimizer=Adam(learning_rate=0.0025), loss='mse')

# Create a learning rate scheduler
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=5, verbose=2, mode='min',
    min_delta=0.0001, cooldown=0, min_lr=0
)
early_stopping = EarlyStopping(monitor="loss", patience=10)
# Train the autoencoder
autoencoder.fit(
    x_train_pca, 
    x_train_pca, 
    epochs=500, 
    batch_size=256, 
    shuffle=True,
    callbacks=[lr_scheduler, early_stopping],
    validation_data=(x_test_pca, x_test_pca), 
    verbose=2)

# Create binary hash codes
x_train_encoded = encoder.predict(x_train_pca)
x_test_encoded = encoder.predict(x_test_pca)
x_train_binary = np.where(x_train_encoded > threshold, 1, 0)
x_test_binary = np.where(x_test_encoded > threshold, 1, 0)

# Reconstruct the images using the binary hash codes
x_train_reconstructed = autoencoder.predict(x_train_pca)
x_test_reconstructed = autoencoder.predict(x_test_pca)

# Calculate the MSE between original and reconstructed images
mse_train = np.mean(np.square(x_train_pca - x_train_reconstructed))
mse_test = np.mean(np.square(x_test_pca - x_test_reconstructed))

print(f"Training MSE: {mse_train:.4f}")
print(f"Testing MSE: {mse_test:.4f}")

# Inverse PCA transformation to obtain the original image shape
x_train_reconstructed_inv = pca.inverse_transform(x_train_reconstructed)
x_test_reconstructed_inv = pca.inverse_transform(x_test_reconstructed)

# Reshape the reconstructed images
x_train_reconstructed_inv = x_train_reconstructed_inv.reshape((x_train_reconstructed_inv.shape[0], 32, 32, 3))
x_test_reconstructed_inv = x_test_reconstructed_inv.reshape((x_test_reconstructed_inv.shape[0], 32, 32, 3))

# Visualize the original and reconstructed images
def visualize_images(original, reconstructed, num_images=20):
    fig, axes = plt.subplots(2, num_images, figsize=(15, 4))
    
    for i in range(num_images):
        # Display original images
        ax = axes[0, i]
        ax.imshow(np.clip(original[i], 0, 1))
        ax.axis('off')
        
        # Display reconstructed images
        ax = axes[1, i]
        ax.imshow(np.clip(reconstructed[i], 0, 1))
        ax.axis('off')

    plt.show()

visualize_images(x_train[:20].reshape(-1, 32, 32, 3), x_train_reconstructed_inv[:20])



