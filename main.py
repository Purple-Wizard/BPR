import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import glob
from torchvision import transforms as T
from torchvision.datasets import DatasetFolder
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import EarlyStopping
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image


tiny_imagenet_path = 'dataset/tiny-imagenet-200'
img_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(
    tiny_imagenet_path) for f in filenames if f.endswith('.JPEG')]

transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

dataset = CustomDataset(img_paths, transform=transform)
train_data, val_data = train_test_split(
    dataset, test_size=0.2, random_state=42)

# Create DataLoaders for training and validation sets
train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)

# Get numpy arrays from the DataLoaders
train_images, train_labels = next(iter(train_loader))
val_images, val_labels = next(iter(val_loader))

# Preprocessing
train_images_2d = train_images.view(-1, 64 * 64 * 3).numpy()
val_images_2d = val_images.view(-1, 64 * 64 * 3).numpy()

# Apply PCA
n_components = 500
pca = PCA(n_components=n_components)
train_images_pca = pca.fit_transform(train_images_2d)
val_images_pca = pca.transform(val_images_2d)

# Scale the PCA components
scaler = MinMaxScaler()
train_images_pca_norm = scaler.fit_transform(train_images_pca)
val_images_pca_norm = scaler.transform(val_images_pca)

# Create a binary autoencoder
input_dim = n_components
encoding_dim = 250

input_layer = Input(shape=(input_dim,))
encoder_layer = Dense(encoding_dim, activation="sigmoid")(input_layer)
decoder_layer = Dense(input_dim, activation="sigmoid")(encoder_layer)

autoencoder = Model(input_layer, decoder_layer)
encoder = Model(input_layer, encoder_layer)

# Custom binary crossentropy loss


def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


# Compile the autoencoder model
autoencoder.compile(optimizer=Adam(learning_rate=0.001),
                    loss=binary_crossentropy)

# Add early stopping
early_stopping = EarlyStopping(monitor="val_loss", patience=5)

# Train the autoencoder
autoencoder.fit(
    train_images_pca_norm,
    train_images_pca_norm,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(val_images_pca_norm, val_images_pca_norm),
    callbacks=[early_stopping],
    verbose=0
)

# Choose an image
image_index = 0
original_image = train_images[image_index].numpy().transpose(1, 2, 0)
image_2d = train_images_pca_norm[image_index].reshape(1, -1)

# Encode and decode the image using the binary autoencoder
encoded_data = encoder.predict(image_2d)
decoded_data = autoencoder.predict(image_2d)

# Inverse scale the decoded data
decoded_data = scaler.inverse_transform(decoded_data)

# Reconstruct the image using PCA
reconstructed_data = pca.inverse_transform(decoded_data)
reconstructed_image = reconstructed_data.reshape(original_image.shape)

# Compute the similarity measure (Mean Squared Error)
mse = mean_squared_error(original_image, reconstructed_image)

# Print the results
print(f"Original Image Shape: {original_image.shape}")
print(f"PCA Data Shape: {image_2d.shape}")
print(f"Encoded Data Shape: {encoded_data.shape}")
print(f"Reconstructed Image Shape: {reconstructed_image.shape}")
print(f"Mean Squared Error: {mse}")

# Visualize the original and reconstructed images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(original_image, cmap='gray')
axes[0].set_title("Original Image")
axes[1].imshow(reconstructed_image, cmap='gray')
axes[1].set_title("Reconstructed Image")
plt.show()
