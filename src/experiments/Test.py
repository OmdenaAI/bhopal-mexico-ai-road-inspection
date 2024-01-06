import ee 
import numpy as np
import pandas as pd
import tensorflow as tf
from arcgis.gis import GIS
from arcgis.mapping import WebMap
from arcgis.features import FeatureLayer, FeatureSet
from sklearn.decomposition import PCA
import geemap
import cv2

service_account = ''
credentials = ee.ServiceAccountCredentials(service_account, '')
ee.Initialize(credentials)

# Method 1 : Using Google earth Engine and ArcGIS ONLINE 
# ISSUE : ROad layers are owned by various organisations or government , so for arcGIS the problem is that we will have to ask for the access for the road layer with the credintials to move forward
image = ee.Image('LANDSAT/LC08/C01/T1_TOA/LC08_123032_20140515')

[[77.50023668318029, 28.476423038130335],
          [77.50184600858923, 28.474800929705655],
          [77.5043994715714, 28.47640417654759],
          [77.50437801389928, 28.477234083001576],
          [77.5040561488175, 28.478271456899524],
          [77.5029403498673, 28.479742624229637],
          [77.50126665144201, 28.480421617624227],
          [77.498562984755, 28.479025904244033]]

roi = ee.Geometry.Polygon([
    [[77.50023668318029, 28.476423038130335],
          [77.50184600858923, 28.474800929705655],
          [77.5043994715714, 28.47640417654759],
          [77.50437801389928, 28.477234083001576],
          [77.5040561488175, 28.478271456899524],
          [77.5029403498673, 28.479742624229637],
          [77.50126665144201, 28.480421617624227],
          [77.498562984755, 28.479025904244033]]
])

image_roi = image.clip(roi)
layer_url = 'https://services.arcgis.com/...'
road_network_layer = FeatureLayer(layer_url)
query = road_network_layer.query()
feature_set = query.features

# Create a binary road map
road_map = np.zeros_like(image_roi)
for feature in feature_set:
    # Convert the feature to a numpy array
    array = np.array(feature.geometry.getInfo()['coordinates'])
    
    # Convert the coordinates to pixel coordinates
    coords = np.round(image_roi.projection().getFractionalCoordinates(array)).astype(int)
    
    # Draw a line on the road map
    for i in range(coords.shape[0] - 1):
        road_map = cv2.line(road_map, tuple(coords[i]), tuple(coords[i+1]), color=255, thickness=5)

# Perform principal component analysis to reduce dimensionality
pca = PCA(n_components=3)
pca.fit(image_roi)
image_pca = pca.transform(image_roi)

# Normalize the image
image_pca_norm = (image_pca - image_pca.min()) / (image_pca.max() - image_pca.min())


#METHOD 2 
#USING GMAPS API 
#ISSUE : 200$ worth of API CALLS ON FREE ACCESS
import cv2
import numpy as np
import tensorflow as tf
from googlemaps import Client
from io import BytesIO
from PIL import Image
import requests
# Set up Google Maps API client
gmaps = Client(api_key='YOUR_API_KEY')

# Get satellite image of an area using Google Maps API
area = '' # Enter the name of the area you want to capture
zoom =  '' # Set the zoom level
size = (640, 640) # Set the size of the image
location = gmaps.geocode(area)[0]['geometry']['location']
lat, lng = location['lat'], location['lng']
img_url = f''
response = requests.get(img_url)
img = np.array(Image.open(BytesIO(response.content)))

# Mark the roads on the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

# Convert to binary mapping
ret, binary_map = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
