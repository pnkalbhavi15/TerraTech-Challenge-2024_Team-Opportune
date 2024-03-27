let map;
let heatmap;
let cropCoordinatesMap = {}; // To store preprocessed coordinates

function initMap() {
  map = L.map("map").setView([20.5937, 78.9629], 4);

  // Load GeoJSON.
  fetch("static/coordinates_with_gender.geojson")
    .then((response) => response.json())
    .then((data) => {
      L.geoJSON(data, {
        pointToLayer: function (feature, latlng) {
          return L.circleMarker(latlng, {
            fillColor: "black",
            fillOpacity: 0.8,
            weight: 0,
            radius: 3, // Adjust the size of the marker
          });
        },
        onEachFeature: function (feature, layer) {
          layer.on("click", function (event) {
            const pincode = feature.properties.pincode;
            // Redirect to the results page for the clicked pincode
            window.location.href = "/search?pincode=" + pincode;
          });
        },
      }).addTo(map);
    })
    .catch((error) => {
      console.error("Error loading GeoJSON:", error);
    });

  // Create markers based on gender filter.
  const genderFilter = document.getElementById("genderFilter");
  genderFilter.addEventListener("change", function () {
    const selectedGender = genderFilter.value;
    map.eachLayer(function (layer) {
      if (layer instanceof L.CircleMarker) {
        const gender = layer.feature.properties.gender;
        if (selectedGender === "all" || gender === selectedGender) {
          layer.setStyle({ opacity: 1 }); // Show the marker
        } else {
          layer.setStyle({ opacity: 0 }); // Hide the marker
        }
      }
    });
  });

  // Initialize the heatmap
  heatmap = L.heatLayer([], {
    radius: 20, // Adjust the radius of the heatmap points
    blur: 15, // Adjust the blur of the heatmap layer
    gradient: { 0.4: "green", 0.6: "yellow", 1: "red" }, // Define the gradient colors from green to red
  });
}

// Define the initMap function globally
window.initMap = initMap;

// Load Leaflet and Leaflet.heat libraries asynchronously
const leafletScript = document.createElement("script");
leafletScript.src = "https://unpkg.com/leaflet/dist/leaflet.js";
leafletScript.onload = function () {
  const heatScript = document.createElement("script");
  heatScript.src = "https://unpkg.com/leaflet.heat/dist/leaflet-heat.js";
  heatScript.onload = function () {
    initMap();
  };
  document.head.appendChild(heatScript);
};
document.head.appendChild(leafletScript);

function toggleHeatmap() {
  const selectedCrop = document.getElementById("cropSelect").value;

  if (selectedCrop === "none") {
    map.removeLayer(heatmap);
  } else {
    if (cropCoordinatesMap[selectedCrop]) {
      Promise.all(cropCoordinatesMap[selectedCrop].map(fetchLatLngForPincode))
        .then((coordinates) => {
          const validCoordinates = coordinates.filter(
            (coord) => coord !== null
          );
          heatmap.setLatLngs(validCoordinates);
          map.addLayer(heatmap);
        })
        .catch((error) => {
          console.error("Error fetching coordinates:", error);
        });
    } else {
      alert("Coordinates not available for selected crop.");
    }
  }
}

function fetchLatLngForPincode(pincode) {
  const url = `https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${pincode}`;

  return fetch(url)
    .then((response) => {
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      return response.json();
    })
    .then((data) => {
      if (data && data.length > 0) {
        const location = data[0];
        return [location.lat, location.lon];
      } else {
        return null;
      }
    })
    .catch((error) => {
      console.error("Error fetching coordinates:", error);
      return null;
    });
}

function fetchPopulationData(pincode) {
  // Fetch population data for the given pincode
  // You need to implement this function based on your dataset
  // This function should return the population value for the given pincode
  // If no data is available, return null
}

function updateHeatmap() {
  // This function is not needed for Leaflet implementation as the heatmap is updated dynamically.
}
