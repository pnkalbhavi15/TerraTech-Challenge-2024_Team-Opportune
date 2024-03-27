import pandas as pd
import requests
import urllib.parse

# Load dataset into a pandas DataFrame
df = pd.read_csv('cropMain20240316060379.csv')

# Filter DataFrame to include only relevant columns
bc_subset_df = df[['Name of BC', 'Pincode']]

# Select a subset of rows for trial conversion
trial_subset_df = bc_subset_df.head(3012)  # Adjust the number of rows as needed for your trial

# Define your Google Maps Geocoding API key
GOOGLE_MAPS_API_KEY = 'AIzaSyAvjxep2-MeoCrjbrIfRJzzw58450jmygc'

def get_lat_lng(pincode):
    url = f'https://maps.googleapis.com/maps/api/geocode/json?address={urllib.parse.quote(str(pincode))}&key={GOOGLE_MAPS_API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'OK':
            location = data['results'][0]['geometry']['location']
            return location['lat'], location['lng']
    return None, None

# Convert pincodes to latitude and longitude
coordinates = []
for index, row in trial_subset_df.iterrows():
    pincode = row['Pincode']
    lat, lng = get_lat_lng(pincode)
    if lat is not None and lng is not None:
        coordinates.append({'Pincode': pincode, 'Latitude': lat, 'Longitude': lng})

# Create a DataFrame with the coordinates
coordinates_df = pd.DataFrame(coordinates)

# Save DataFrame to a new CSV file
coordinates_df.to_csv('pincodes_coordinates.csv', index=False)
