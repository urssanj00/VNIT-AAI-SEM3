import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage import data
import numpy as np

# Load and normalize the grayscale image
image = data.camera()  # shape: (512, 512), grayscale
X = image / 255.0      # Normalize pixel values to [0, 1]

# Function to compress and reconstruct the image using PCA
def pca_image_compression(X, n_components):
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)
    return X_reconstructed

# List of PCA component counts to test
components_list = [5, 20, 50]

# Total plots = 1 original + len(components_list)
plt.figure(figsize=(15, 4))

# Plot original image
plt.subplot(1, len(components_list) + 1, 1)
plt.imshow(X, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Plot reconstructed images
for i, n in enumerate(components_list):
    reconstructed = pca_image_compression(X, n)
    plt.subplot(1, len(components_list) + 1, i + 2)
    plt.imshow(reconstructed, cmap='gray')
    plt.title(f'{n} Components')
    plt.axis('off')

plt.suptitle("Original vs Reconstructed Images using PCA")
plt.tight_layout()
plt.savefig('problem_2.png')
