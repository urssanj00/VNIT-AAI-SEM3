import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def variance_analysis(data_set, plot_name):
    print(f'dataset shape {data_set.data.shape}')
    X = data_set.data
    y = data_set.target

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA()
    pca.fit(X_scaled)

    # Calculate explained variance ratios and cumulative variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Find the minimum number of components to retain 95% variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

    # Plot cumulative variance vs. number of components
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='b')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance Threshold')
    plt.axvline(x=n_components_95, color='g', linestyle='--', label=f'Min Components: {n_components_95}')
    plt.title('Cumulative Explained Variance vs. Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_name)

    # Print results
    print(f"Explained Variance Ratios: {explained_variance_ratio}")
    print(f"Cumulative Variance: {cumulative_variance}")
    print(f"Minimum number of components to retain 95% variance: {n_components_95}")

# Load and preprocess the Wine dataset
print(f'Analysing Wine Dataset')
data_w = load_wine()
variance_analysis(data_w, "pca_win.png")
print()
print(f'===============================================================')
print(f'Analysing Breast Cancer Dataset')
data_b = load_breast_cancer()
variance_analysis(data_b, "pca_breast_cancer.png")
