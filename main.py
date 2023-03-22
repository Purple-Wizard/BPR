import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
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
        image = Image.open(img_path).convert('RGB')
        label = os.path.split(os.path.dirname(img_path))[-1]

        if self.transform:
            image = self.transform(image)

        return image, label

tiny_imagenet_path = 'dataset/tiny-imagenet-200'
img_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(
    tiny_imagenet_path) for f in filenames if f.endswith('.JPEG')]

transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor()
])

dataset = CustomDataset(img_paths, transform=transform)

# Create DataLoader for the entire dataset
data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

# Get numpy arrays from the DataLoader
images, labels = next(iter(data_loader))

# Preprocessing
images_2d = images.view(-1, 64 * 64 * 3).numpy()

# Apply PCA
n_components = 1000
pca = PCA(n_components=n_components)
images_pca = pca.fit_transform(images_2d)

# Scale the PCA components
scaler = MinMaxScaler()
images_pca_norm = scaler.fit_transform(images_pca)

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
early_stopping = EarlyStopping(monitor="loss", patience=5)

# Train the autoencoder
autoencoder.fit(
    images_pca_norm,
    images_pca_norm,
    epochs=50,
    batch_size=256,
    shuffle=True,
    callbacks=[early_stopping],
    verbose=0
)

# Choose an image
image_index = 0
original_image = images[image_index].numpy().transpose(1, 2, 0)
image_2d = images_pca_norm[image_index].reshape(1, -1)

# Encode and decode the image using the binary autoencoder
encoded_data = encoder.predict(image_2d)
decoded_data = autoencoder.predict(image_2d)

# Inverse scale the decoded data
decoded_data = scaler.inverse_transform(decoded_data)

# Reconstruct the image using PCA
reconstructed_data = pca.inverse_transform(decoded_data)
reconstructed_image = reconstructed_data.reshape(original_image.shape)

# Compute the similarity measure (Mean Squared Error)
mse = mean_squared_error(original_image.reshape(-1), reconstructed_image.reshape(-1))

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
