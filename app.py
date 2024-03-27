from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import json
import pandas as pd
import os
if not os.path.exists('tmp'):
    os.makedirs('tmp')
import geopandas as gpd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import folium
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import plotly.express as px
app = Flask(__name__, static_url_path='/static')

# Get the current directory path
current_dir = os.path.dirname(__file__)

# Load the dataset into a pandas DataFrame
df = pd.read_csv('cropMain20240316060379.csv')

# Load pincode coordinates from the generated CSV file
pincode_coordinates_df = pd.read_csv('pincodes_coordinates.csv')

# Merge pincode coordinates with the original DataFrame
bc_info_df = pd.merge(df, pincode_coordinates_df, on='Pincode', how='left')

# Filter DataFrame to include only relevant columns
bc_info_df = bc_info_df[['Name of BC', 'Contact Number', 'Gender', 'Bank Name', 'State', 'District ', 'Office Name', 'Pincode', 'Corporate BC name', 'ODOP product','CODE WORD OF PRODUCT LIST', 'CROP1 NAME','CROP1 SEASON','CROP1 AREA','CROP1 PRODUCTION','CROP1 Pincode','CROP2 NAME','CROP2 SEASON','CROP2 AREA','CROP2 PRODUCTION','CROP3 NAME','CROP3 SEASON','CROP3 AREA','CROP3 PRODUCTION']]

# Extract unique crop names from the "CROP NAME" column
unique_crops = df['CROP1 NAME'].unique().tolist()

# Convert the list of unique crop names into JSON format
crop_data_json = json.dumps(unique_crops)

# Load population dataset
population_data = pd.read_csv("population_data.csv")

# Split the combined column into separate columns for District Name and State Letter Code
population_data[['District Name', 'State Letter Code']] = population_data['District Name, State Letter Code'].str.split(', ', expand=True)

# Load shapefile
district_boundaries = gpd.read_file("IND_adm2.shp")

# Merge population data with shapefile
merged_data = district_boundaries.merge(population_data, how='left', left_on='NAME_2', right_on='District Name')


# Read cropMain.csv for Crop1
crop_data1 = pd.read_csv(os.path.join(current_dir, 'cropMain.csv'))
coordinates_data1 = pd.read_csv(os.path.join(current_dir, 'pincodes_coordinates.csv'))
merged_data1 = pd.merge(crop_data1, coordinates_data1, on='Pincode', how='inner')
extracted_data1 = merged_data1[['CROP1 NAME', 'Pincode', 'Latitude', 'Longitude']].copy()
extracted_data1.dropna(inplace=True)
crop_options1 = extracted_data1['CROP1 NAME'].unique()

# Read cropMain.csv for Crop2
crop_data2 = pd.read_csv(os.path.join(current_dir, 'cropMain.csv'))
coordinates_data2 = pd.read_csv(os.path.join(current_dir, 'pincodes_coordinates.csv'))
merged_data2 = pd.merge(crop_data2, coordinates_data2, on='Pincode', how='inner')
extracted_data2 = merged_data2[['CROP2 NAME', 'Pincode', 'Latitude', 'Longitude']].copy()
extracted_data2.dropna(inplace=True)
crop_options2 = extracted_data2['CROP2 NAME'].unique()

# Read cropMain.csv for Crop3
crop_data3 = pd.read_csv(os.path.join(current_dir, 'cropMain.csv'))
coordinates_data3 = pd.read_csv(os.path.join(current_dir, 'pincodes_coordinates.csv'))
merged_data3 = pd.merge(crop_data3, coordinates_data3, on='Pincode', how='inner')
extracted_data3 = merged_data3[['CROP3 NAME', 'Pincode', 'Latitude', 'Longitude']].copy()
extracted_data3.dropna(inplace=True)
crop_options3 = extracted_data3['CROP3 NAME'].unique()

gender_data = pd.read_csv(os.path.join(current_dir, 'cropMain.csv'))
coordinates_data21 = pd.read_csv(os.path.join(current_dir, 'pincodes_coordinates.csv'))
merged_data21 = pd.merge(gender_data, coordinates_data21, on='Pincode', how='inner')
extracted_data21 = merged_data21[['Gender', 'Pincode', 'Latitude', 'Longitude']].copy()  # Make a copy to avoid SettingWithCopyWarning
extracted_data21.dropna(inplace=True)
gender_options = extracted_data21['Gender'].unique()

State_data31 = pd.read_csv(os.path.join(current_dir, 'in.csv'))
coordinates_data31 = pd.read_csv(os.path.join(current_dir, 'crop_yield.csv'))
merged_data31 = pd.merge(State_data31, coordinates_data31, on='State', how='inner')
extracted_data31 = merged_data31[['Fertilizer', 'State', 'Latitude', 'Longitude']].copy()  
extracted_data31.dropna(inplace=True)
State_options31 = extracted_data31['State'].unique()

State_data32 = pd.read_csv(os.path.join(current_dir, 'in.csv'))
coordinates_data32 = pd.read_csv(os.path.join(current_dir, 'crop_yield.csv'))
merged_data32 = pd.merge(State_data32, coordinates_data32, on='State', how='inner')
extracted_data32 = merged_data32[['Pesticide', 'State', 'Latitude', 'Longitude']].copy()  
extracted_data32.dropna(inplace=True)
State_options32 = extracted_data32['State'].unique()

State_data33 = pd.read_csv(os.path.join(current_dir, 'in.csv'))
coordinates_data33 = pd.read_csv(os.path.join(current_dir, 'crop_yield.csv'))
merged_data33 = pd.merge(State_data33, coordinates_data33, on='State', how='inner')
extracted_data33 = merged_data33[['Yield', 'State', 'Latitude', 'Longitude']].copy()  
extracted_data33.dropna(inplace=True)
State_options33 = extracted_data33['State'].unique()

district_data4 = pd.read_csv(os.path.join(current_dir, 'Lit_Long_lati.csv'))
literacy_data4 = pd.read_csv(os.path.join(current_dir, 'Literacy Data 2011.csv'))
merged_data4 = pd.merge(district_data4, literacy_data4, on='District', how='inner')
district_options4 = merged_data4['District'].unique()

crop_data5 = pd.read_csv(os.path.join(current_dir, 'cropMain.csv'))
coordinates_data5 = pd.read_csv(os.path.join(current_dir, 'pincodes_coordinates.csv'))
merged_data5 = pd.merge(crop_data5, coordinates_data5, on='Pincode', how='inner')
extracted_data5 = merged_data5[['Bank Name', 'Pincode', 'Latitude', 'Longitude']].copy()  # Make a copy to avoid SettingWithCopyWarning
extracted_data5.dropna(inplace=True)
crop_options5 = extracted_data5['Bank Name'].unique()


# Load data from CSV file
file_path = 'prices.csv'
data = pd.read_csv(file_path)

# Load the dataset
education_data = pd.read_csv('census_edu.csv')

# Function to plot cluster maps for districts in a given state
def plot_cluster_maps(state_code):
    # Filter data by state code
    state_data = education_data[education_data['State Code'] == state_code]
    
    # Group data by town name
    grouped_data = state_data.groupby('Town Name')

    # Choose the optimal number of clusters based on the elbow method
    optimal_k = 3  # You can adjust this based on the elbow method or any other criteria

    # Create a BytesIO object to hold the plot image
    img_bytes_io = BytesIO()

    # Iterate over each town name and plot cluster map
    for town_name, data in grouped_data:
        # Select relevant features for clustering
        features = data[['Illiterate Persons', 'Literate Persons']]  # Update features as needed

        # Data preprocessing: Handling missing values and normalization if necessary
        features.dropna(inplace=True)

        # Perform KMeans clustering with the optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(features)

        # Plot the cluster map
        plt.figure()
        plt.scatter(features['Illiterate Persons'], features['Literate Persons'], c=clusters, cmap='viridis', marker='o')
        plt.title(f'({town_name})')
        plt.xlabel('Illiterate')
        plt.ylabel('Literate')
        plt.savefig(img_bytes_io, format='png')  # Save plot to BytesIO object
        plt.close()

    # Return BytesIO object containing the plot images
    return img_bytes_io.getvalue()

# Function to generate heatmap and return base64 encoded image
def generate_heatmap():
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    merged_data.plot(column='District Population', cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
    ax.set_title('Population Heatmap by District')
    plt.axis('off')  # Turn off axis
    # Save plot to bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    # Encode image to base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)  # Close plot to free memory
    return image_base64

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/index')
def index():
    # Render the template containing the navigation bar and map
    return render_template('index.html', crop_data=crop_data_json)

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        pincode = request.form.get('pincode')
    elif request.method == 'GET':
        pincode = request.args.get('pincode')

    if pincode is None or pincode == '':
        # Handle case where pincode is not provided
        return render_template('index.html', error_message="Please enter a pincode")

    # Filter data based on the provided pincode
    filtered_data = bc_info_df[bc_info_df['Pincode'] == int(pincode)]

    if filtered_data.empty:
        # Handle case where no data is found for the provided pincode
        return render_template('index.html', error_message=f"No data found for pincode: {pincode}")

    # Convert filtered data to HTML table
    table = filtered_data.to_html(classes='data', header="true", index=False)
    
    # Render the results template with the pincode and table
    return render_template('results.html', pincode=pincode, table=table)


@app.route('/index1')
def index1():
    return render_template('index1.html', crop_options=crop_options1)

@app.route('/index2')
def index2():
    return render_template('index2.html', crop_options=crop_options2)

@app.route('/index3')
def index3():
    return render_template('index3.html', crop_options=crop_options3)

@app.route('/index4')
def index4():
    return render_template('index4.html', district_options=district_options4)

@app.route('/bank')
def bank():
    return render_template('bank.html', crop_options=crop_options5)

@app.route('/index21')
def index21():
    return render_template('index21.html', gender_options=gender_options)

@app.route('/index31')
def index31():
    return render_template('index31.html', State_options=State_options31)

@app.route('/index32')
def index32():
    return render_template('index32.html', State_options=State_options32)

@app.route('/index33')
def index33():
    return render_template('index33.html', State_options=State_options33)

@app.route('/update_map1', methods=['POST'])
def update_map1():
    selected_crop = request.form['crop']
    if selected_crop == 'all':
        filtered_data = extracted_data1
    else:
        filtered_data = extracted_data1[extracted_data1['CROP1 NAME'] == selected_crop]

    # Create a list of latitude and longitude coordinates
    heat_map_data = [[row['Latitude'], row['Longitude']] for index, row in filtered_data.iterrows()]

    return jsonify(heat_map_data)

@app.route('/update_map2', methods=['POST'])
def update_map2():
    selected_crop = request.form['crop']
    if selected_crop == 'all':
        filtered_data = extracted_data2
    else:
        filtered_data = extracted_data2[extracted_data2['CROP2 NAME'] == selected_crop]

    # Create a list of latitude and longitude coordinates
    heat_map_data = [[row['Latitude'], row['Longitude']] for index, row in filtered_data.iterrows()]

    return jsonify(heat_map_data)

@app.route('/update_map3', methods=['POST'])
def update_map3():
    selected_crop = request.form['crop']
    if selected_crop == 'all':
        filtered_data = extracted_data3
    else:
        filtered_data = extracted_data3[extracted_data3['CROP3 NAME'] == selected_crop]

    # Create a list of latitude and longitude coordinates
    heat_map_data = [[row['Latitude'], row['Longitude']] for index, row in filtered_data.iterrows()]

    return jsonify(heat_map_data)

@app.route('/update_map4', methods=['POST'])
def update_map4():
    selected_district = request.form['District']  
    if selected_district == 'all':
        filtered_data = merged_data4
    else:
        filtered_data = merged_data4[merged_data4['District'] == selected_district]

    heat_map_data = [[row['Latitude'], row['Longitude']] for index, row in filtered_data.iterrows()]

    return jsonify(heat_map_data)

@app.route('/update_map5', methods=['POST'])
def update_map5():
    selected_crop = request.form['crop']
    if selected_crop == 'all':
        filtered_data = extracted_data5
    else:
        filtered_data = extracted_data5[extracted_data5['Bank Name'] == selected_crop]
    heat_map_data = [[row['Latitude'], row['Longitude']] for index, row in filtered_data.iterrows()]

    return jsonify(heat_map_data)

@app.route('/update_map21', methods=['POST'])
def update_map21():
    selected_gender = request.form['gender']
    if selected_gender == 'all':
        filtered_data = extracted_data21
    else:
        filtered_data = extracted_data21[extracted_data21['Gender'] == selected_gender]

    # Create a list of latitude and longitude coordinates
    heat_map_data = [[row['Latitude'], row['Longitude']] for index, row in filtered_data.iterrows()]

    return jsonify(heat_map_data)

@app.route('/update_map31', methods=['POST'])
def update_map31():
    selected_state = request.form['State']  
    if selected_state == 'all':
        filtered_data = extracted_data31
    else:
        filtered_data = extracted_data31[extracted_data31['State'] == selected_state]

    heat_map_data = [[row['Latitude'], row['Longitude']] for index, row in filtered_data.iterrows()]

    return jsonify(heat_map_data)

@app.route('/update_map32', methods=['POST'])
def update_map32():
    selected_state = request.form['State']  
    if selected_state == 'all':
        filtered_data = extracted_data32
    else:
        filtered_data = extracted_data32[extracted_data32['State'] == selected_state]

    heat_map_data = [[row['Latitude'], row['Longitude']] for index, row in filtered_data.iterrows()]

    return jsonify(heat_map_data)

@app.route('/update_map33', methods=['POST'])
def update_map33():
    selected_state = request.form['State']  
    if selected_state == 'all':
        filtered_data = extracted_data33
    else:
        filtered_data = extracted_data33[extracted_data33['State'] == selected_state]

    heat_map_data = [[row['Latitude'], row['Longitude']] for index, row in filtered_data.iterrows()]

    return jsonify(heat_map_data)

@app.route('/generate_population_heatmap')
def generate_population_heatmap():
    heatmap_image = generate_heatmap()
    return render_template('population_heatmap.html', heatmap_image=heatmap_image)

@app.route('/static/coordinates_with_gender.geojson')
def serve_geojson():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(os.path.join(root_dir, 'static'), 'coordinates_with_gender.geojson')

@app.route('/literacy_chloropleth')
def literacy_chloropleth():
    return render_template('literacy_chloropleth.html')

@app.route('/map', methods=['POST'])
def update_map():
    # Get the selected options from the form
    options = request.form.getlist('option')

    # Load the state shapefile
    shapefile_path = "IND_adm1.shp"
    india_states = gpd.read_file(shapefile_path)

    # Load the literacy dataset
    literacy_data = pd.read_csv("GOI.csv")

    # Create a Folium map
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

    for option in options:
        # Add choropleth layer for each selected option
        folium.Choropleth(
            geo_data=india_states,
            name=option,
            data=literacy_data,
            columns=['Country/ States/ Union Territories Name', option],
            key_on='feature.properties.NAME_1',
            fill_color='YlGnBu',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Literacy Rate (%)'
        ).add_to(m)

    folium.LayerControl().add_to(m)

    # Save the map as HTML
    map_html = m.get_root().render()

    return render_template('map.html', map_html=map_html)

@app.route('/all_chloropleth')
def all_chloropleth():
    district_data = pd.read_csv("all.csv")
    columns =  columns = [col for col in district_data.columns.tolist() if col not in ['Unnamed: 0', 'State', 'District']]
    return render_template('all_chloropleth.html', columns=columns)

@app.route('/map_all_chloropleth', methods=['POST'])
def map_all_chloropleth():
    district_data = pd.read_csv("all.csv")
    selected_columns = request.form.getlist('selected_columns')
    selected_value = request.form['value']

    india_districts = gpd.read_file('IND_adm2.shp')

    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

    for column in selected_columns:
        filtered_data = district_data[district_data[column] == selected_value]
        merged_data = india_districts.merge(filtered_data, how='left', left_on='NAME_2', right_on='District')
        merged_data.dropna(subset=['District'], inplace=True)

        folium.Choropleth(
            geo_data=merged_data,
            name=f'choropleth_{column}',
            data=merged_data,
            columns=['District', column],
            key_on='feature.properties.NAME_2',
            fill_color='YlOrRd',  # Change to a different color scheme, such as Yellow-Orange-Red
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=column
        ).add_to(m)

    folium.LayerControl().add_to(m)

    map_html = m.get_root().render()

    return render_template('map_all_chloropleth.html', map_html=map_html)

@app.route('/market_price_analysis')
def market_price_analysis():
    # Run R script
    subprocess.run(['Rscript', 'main.r'])
    
    return render_template('market_price_analysis.html')

@app.route('/plot/<plot_name>')
def plot(plot_name):
    # Serve dynamically generated plot
    return send_file(f'tmp/{plot_name}.png', mimetype='image/png')

    
df = pd.read_csv("market_.csv")

def occurrences_of_states():
    state_counts = df['State'].value_counts()
    plt.figure(figsize=(10, 6))
    state_counts.plot(kind='bar', color="red")
    plt.title('Occurrences of States in the Dataset')
    plt.xlabel('State')
    plt.ylabel('Count')
    plt.xticks(rotation=90)  
    plt.savefig('static/state_occurrences.png')  # Save the plot as an image
    return 'static/state_occurrences.png'

def crop_types_in_each_state():
    state_crop_counts = df.groupby(['State', 'CROP1 NAME']).size().unstack(fill_value=0)

    plt.figure(figsize=(15, 10))
    state_crop_counts.plot(kind='bar', stacked=True)
    plt.title('Crop Types in Each State')
    plt.xlabel('State')
    plt.ylabel('Count')
    plt.xticks(rotation=90)  
    plt.legend(title='Crop Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('static/state_crop_types.png')  # Save the plot as an image
    return 'static/state_crop_types.png'

def crop_frequencies_by_state():
    unique_states = df['State'].unique()
    num_rows = int(np.ceil(len(unique_states) / 3))
    fig, axs = plt.subplots(num_rows, 3, figsize=(15, num_rows * 4))
    axs = axs.flatten()
    for i, state in enumerate(unique_states):
        state_df = df[df['State'] == state]

        unique_crops = state_df['CROP1 NAME'].unique()

        axs[i].hist(state_df['CROP1 NAME'], bins=len(unique_crops), color='red', edgecolor='black')

        axs[i].set_xticks(range(len(unique_crops)))
        axs[i].set_xticklabels(unique_crops, rotation=45, ha='right')

        axs[i].set_title(f'Crop Frequencies in {state}')
        axs[i].set_xlabel('Crop Name')
        axs[i].set_ylabel('Frequency')
        axs[i].grid(axis='y')

    for j in range(len(unique_states), len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.savefig('static/crop_frequencies_by_state.png')  # Save the plot as an image
    return 'static/crop_frequencies_by_state.png'

def plot_production_by_pincode(state_name, crop_name):
    # Filter data based on user input
    filtered_df = df[(df['State'] == state_name) & (df['CROP1 NAME'] == crop_name)]

    pincode_production = filtered_df.groupby('Pincode')['CROP1 PRODUCTION'].sum()

    if pincode_production.empty:
        print("No data found for the specified state and crop combination.")
        return None, None

    plt.figure(figsize=(10, 6))
    plt.pie(pincode_production, labels=pincode_production.index, autopct='%1.1f%%')
    plt.title(f'Production of {crop_name} by Pincode in {state_name}')
    img_path = 'static/production_by_pincode.png'
    plt.savefig(img_path)  # Save the plot as an image
    plt.close()  # Close the plot to prevent it from being displayed in the Flask app
    return img_path, pincode_production.to_dict()


def lineplot_time_series():
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='CROP1 SEASON', y='CROP1 PRODUCTION', hue='State')
    plt.title('Time-Series of Crop Production')
    plt.xlabel('CROP1 SEASON')
    plt.ylabel('CROP1 PRODUCTION')
    plt.xticks()
    plt.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left')
    img_path = 'static/lineplot_time_series.png'
    plt.savefig(img_path)  # Save the plot as an image
    plt.close()  # Close the plot to prevent it from being displayed in the Flask app
    return img_path

def count_plots_categorical():
    categorical_columns = ['Gender', 'State', 'ODOP product', 'CODE WORD OF PRODUCT LIST', 'CROP1 NAME', 'CROP1 SEASON']

    for col in categorical_columns:
        plt.figure(figsize=(15, 6))
        sns.countplot(data=df, x=col)
        plt.title(f'Frequency of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        img_path = f'static/count_plot_{col}.png'
        plt.savefig(img_path)  # Save the plot as an image
        plt.close()  # Close the plot to prevent it from being displayed in the Flask app

    return [f'static/count_plot_{col}.png' for col in categorical_columns]

@app.route('/index_market')
def index_market():
    return render_template('index_market.html')


@app.route('/state_occurrences')
def state_occurrences():
    img_path = occurrences_of_states()
    return render_template('image_display.html', img_path=img_path)

@app.route('/crop_types_in_each_state')
def crop_types_in_each_state_route():
    img_path = crop_types_in_each_state()
    return render_template('image_display.html', img_path=img_path)

@app.route('/crop_frequencies_by_state')
def crop_frequencies_by_state_route():
    img_path = crop_frequencies_by_state()
    return render_template('image_display.html', img_path=img_path)

@app.route('/production_by_pincode', methods=['GET', 'POST'])
def production_by_pincode():
    if request.method == 'POST':
        state_name = request.form['state']
        crop_name = request.form['crop']
        img_path, pincode_production = plot_production_by_pincode(state_name, crop_name)
        return render_template('production_by_pincode.html', img_path=img_path, pincode_production=pincode_production)

    return render_template('production_by_pincode.html')

@app.route('/count_plots_categorical')
def count_plots_categorical_route():
    img_paths = count_plots_categorical()
    return render_template('image_display_multiple.html', img_paths=img_paths)

@app.route('/lineplot_time_series')
def lineplot_time_series_route():
    img_path = lineplot_time_series()
    return render_template('image_display.html', img_path=img_path)

@app.route('/edu_analysis', methods=['GET', 'POST'])
def edu_analysis():
    if request.method == 'POST':
        state_code = int(request.form['state_code'])
        plot_data = plot_cluster_maps(state_code)
        plot_img = base64.b64encode(plot_data).decode('utf-8')  # Convert image bytes to base64 string
        return render_template('edu_analysis.html', plot_img=plot_img)
    return render_template('edu_analysis.html')

# Load the crop yield data
crop_yield_data = pd.read_csv('agri2.csv')

# Select relevant columns for analysis
crop_yield_data = crop_yield_data[['States', 'Crops', 'Yield (Kg./Hectare) - 2017-18', 'Yield (Kg./Hectare) - 2018-19',
                                   'Yield (Kg./Hectare) - 2019-20', 'Yield (Kg./Hectare) - 2020-21', 'Yield (Kg./Hectare) - 2021-22']]

# Calculate average yield for spatial analysis
crop_yield_data['Average Yield'] = crop_yield_data[['Yield (Kg./Hectare) - 2017-18', 'Yield (Kg./Hectare) - 2018-19',
                                                    'Yield (Kg./Hectare) - 2019-20', 'Yield (Kg./Hectare) - 2020-21',
                                                    'Yield (Kg./Hectare) - 2021-22']].mean(axis=1)

# Sort by average yield for trend analysis
crop_yield_data = crop_yield_data.sort_values(by='Average Yield', ascending=False)

# Load state-wise latitude and longitude data
state_coordinates = pd.read_csv('state_coordinates.csv')  # Update with your actual file name

# Merge the crop yield data with state coordinates
merged_data = pd.merge(crop_yield_data, state_coordinates, on='States')

# Create the scatter mapbox plot
fig = px.scatter_mapbox(merged_data, lat="Latitude", lon="Longitude", hover_name="Crops",
                        color="Average Yield", size="Average Yield",
                        color_continuous_scale=px.colors.sequential.YlGnBu,
                        size_max=15, zoom=3, mapbox_style="carto-positron",
                        title="Spatial Analysis: Crop Yield Across States")

# Convert plot to JSON
scatter_map_json = fig.to_json()

@app.route('/analysis1')
def analysis1():
    return render_template('analysis1.html')

@app.route('/agri_analysis')
def agri_analysis():
    return render_template('agri_analysis.html', scatter_map=scatter_map_json)

from pysal.explore import esda
from pysal.lib import weights
from shapely.geometry import Point

@app.route('/health')
def health():
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

    # Plot spatial distribution of data points
    plt.subplot(1, 2, 1)
    gdf.plot(ax=plt.gca(), color='blue', markersize=5)
    plt.title('Spatial Distribution of Data Points')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Plot significant clusters
    plt.subplot(1, 2, 2)
    gdf.plot(column='doctorCount', cmap='coolwarm', legend=True, ax=plt.gca())
    sig_clusters = gdf[moran_loc.p_sim <= 0.05]
    sig_clusters.plot(color='red', ax=plt.gca(), alpha=0.5, legend=True)
    plt.title('Anselin Local Moran\'s I: Significant Clusters and Outliers for Doctor Count')

    plt.tight_layout()

    # Save plot as an image
    plot_filename = 'static/images/spatial_distribution.png'
    plt.savefig(plot_filename)

    return render_template('health.html', plot_filename=plot_filename)

from flask import Flask, render_template
from moran2 import moran2_bp
from kernel_hosp import kernel_hosp_bp
from cluster_crop import cluster_crop_bp

@app.route('/kernel')
def kernel():
    return render_template('kernel.html')


# Register blueprints
app.register_blueprint(moran2_bp)
app.register_blueprint(kernel_hosp_bp)
app.register_blueprint(cluster_crop_bp)


if __name__ == '__main__':
    app.run(debug=True)
