import pandas as pd
import requests
import json

# Load your dataset
df = pd.read_csv('cropMain20240316060379.csv')

# Function to convert pincode to latitude and longitude and include gender information
def get_lat_long_with_gender(row):
    pincode = row['Pincode']
    gender = row['Gender']
    response = requests.get(f"https://maps.googleapis.com/maps/api/geocode/json?address={pincode}&key=AIzaSyAshA0UPmf_DGu0yXv25Knohs3U_EKqS8k")
    resp_json = response.json()
    if resp_json['status'] == 'OK':
        lat = resp_json['results'][0]['geometry']['location']['lat']
        lng = resp_json['results'][0]['geometry']['location']['lng']
        return {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [lng, lat]
            },
            'properties': {
                'pincode': pincode,
                'gender': gender  # Include gender property
            }
        }
    else:
        return None

# Convert the first 3012 rows and include gender information
features = df.head(3012).apply(get_lat_long_with_gender, axis=1).dropna().tolist()

# Create the GeoJSON structure
geojson = {
    'type': 'FeatureCollection',
    'features': features
}

# Save the GeoJSON to a file
with open('coordinates_with_gender.geojson', 'w') as f:
    json.dump(geojson, f, indent=2)
