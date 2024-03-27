from flask import Blueprint, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns

cluster_crop_bp = Blueprint('cluster_crop', __name__)

@cluster_crop_bp.route('/cluster_crop')
def cluster_crop():
    # Load education data into a pandas DataFrame
    education_data = pd.read_csv("cropMain20240316060379.csv")

    # Drop rows with missing values
    education_data.dropna(inplace=True)

    # Select relevant features for clustering
    education_features = education_data[['CROP2 AREA', 'CROP2 PRODUCTION', 'CROP AREA', 'CROP PRODUCTION']]

    # Check for NaN values after dropping missing values
    if education_features.isnull().values.any():
        return jsonify({"error": "NaN values still present in the dataset. Please check your data cleaning process."}), 400
    else:
        # Standardize the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(education_features)

        # Perform K-means clustering with different numbers of clusters
        inertia_values = []
        silhouette_scores = []
        max_clusters = 10  # maximum number of clusters to try

        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(scaled_features)

            # Calculate inertia (within-cluster sum of squares)
            inertia_values.append(kmeans.inertia_)

            # Calculate silhouette score
            silhouette_scores.append(silhouette_score(scaled_features, kmeans.labels_))

        # Based on the plots, choose the optimal number of clusters
        optimal_num_clusters = 4  # Adjust based on the elbow method and silhouette score analysis

        # Perform K-means clustering with the optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
        kmeans.fit(scaled_features)

        # Add cluster labels to the original DataFrame
        education_data['cluster_label'] = kmeans.labels_

        # Display cluster centers
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        cluster_centers_df = pd.DataFrame(cluster_centers, columns=education_features.columns)
        print("Cluster Centers:")
        print(cluster_centers_df)

        # Create a scatter plot with Seaborn
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=scaled_features[:, 0], y=scaled_features[:, 1], hue=education_data['cluster_label'], palette='Set1', s=100)
        sns.scatterplot(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1], color='black', marker='X', s=200, label='Cluster Centers')
        plt.xlabel('Crop area')
        plt.ylabel('Crop production')
        plt.title('K-means Clustering of crops')
        plt.legend()
        plt.grid(True)
        plt.show()

        return jsonify({"message": "K-means clustering completed successfully."}), 200
