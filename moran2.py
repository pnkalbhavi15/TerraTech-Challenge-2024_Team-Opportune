from flask import Blueprint, jsonify
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pysal.explore import esda
from pysal.lib import weights
from shapely.geometry import Point
import os

moran2_bp = Blueprint('moran2', __name__)

@moran2_bp.route('/moran2')
def moran2():
    # Load CSV data into a DataFrame
    data = pd.read_csv('health_csv.csv')

    # Convert DataFrame to GeoDataFrame
    geometry = [Point(xy) for xy in zip(data['longitude'], data['latitude'])]
    gdf = gpd.GeoDataFrame(data, geometry=geometry)

    # Create a spatial weights matrix (e.g., using k-nearest neighbors with k=5)
    w = weights.KNN.from_dataframe(gdf, k=5)

    # Calculate Anselin Local Moran's I statistic
    moran_loc = esda.Moran_Local(gdf['doctorCount'], w)

    # Plot results
    plt.figure(figsize=(12, 6))
    gdf.plot(ax=plt.gca(), color='blue', markersize=5)
    plt.title('Spatial Distribution of Data Points')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Save plot as an image in the same directory as app.py
    plot_filename = 'moran_output.png'
    plot_path = os.path.join(os.path.dirname(__file__), plot_filename)
    plt.savefig(plot_path)
    plt.show()

    return jsonify({"message": "Spatial distribution plotted successfully.", "filename": plot_filename}), 200
