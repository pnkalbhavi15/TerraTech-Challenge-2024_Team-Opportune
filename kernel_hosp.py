from flask import Blueprint, jsonify, request
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import pandas as pd

kernel_hosp_bp = Blueprint('kernel_hosp', __name__)

@kernel_hosp_bp.route('/kernel_hosp', methods=['POST'])  # Change to POST
def kernel_hosp():
    # Load hospital data
    hospital_data = pd.read_csv("hospital.csv")

    # Drop rows with missing values
    hospital_data.dropna(inplace=True)

    # Select the columns with string data for one-hot encoding
    selected_columns = ['State', 'Hospital', 'Category', 'Specializations']  # Exclude 'Pincode'
    string_data = hospital_data[selected_columns]

    # Perform one-hot encoding
    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(string_data)

    # Convert sparse data to dense numpy array
    encoded_data_dense = encoded_data.toarray()

    # Perform dimensionality reduction using PCA
    pca = PCA(n_components=2)  # Reduce to 2 dimensions
    encoded_data_reduced = pca.fit_transform(encoded_data_dense)

    # Function to perform KDE for a specific state
    def kde_for_state(state):
        # Filter data for the specified state
        state_data = hospital_data[hospital_data['State'] == state]
        if state_data.empty:
            return jsonify({"error": "No data available for the specified state."}), 404

        # Perform one-hot encoding for the filtered data
        state_encoded_data = encoder.transform(state_data[selected_columns])

        # Convert sparse data to dense numpy array
        state_encoded_data_dense = state_encoded_data.toarray()

        # Reduce dimensionality using PCA
        state_encoded_data_reduced = pca.transform(state_encoded_data_dense)

        # Perform kernel density estimation
        kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
        kde.fit(state_encoded_data_reduced)

        # Create a grid for visualization
        x_min, x_max = state_encoded_data_reduced[:, 0].min() - 1, state_encoded_data_reduced[:, 0].max() + 1
        y_min, y_max = state_encoded_data_reduced[:, 1].min() - 1, state_encoded_data_reduced[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Compute KDE values for grid points
        kde_values = np.exp(kde.score_samples(grid_points))
        kde_values = kde_values.reshape(xx.shape)

        # Visualize KDE output
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(xx, yy, kde_values, cmap='viridis')
        plt.colorbar(label='Kernel Density Estimate')
        plt.title('Kernel Density Estimation for Hospitals in {}'.format(state))
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.grid(True)
        plt.savefig('kde_output.png')  # Save plot as an image
        plt.show()

        return jsonify({"message": "Kernel Density Estimation plotted successfully.", "filename": "kde_output.png"}), 200

    # Get state from form data
    state = request.form.get('state')  # Change to request.form.get

    # Perform KDE for the specified state
    return kde_for_state(state)
