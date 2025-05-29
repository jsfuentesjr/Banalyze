#Source Code for Banana Production Output Estimation and Health Status Identification

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from joblib import load
from skimage.feature import hog 

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'images')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the saved models
print("Loading the banana count model...")
banana_count_model_path = os.path.join(os.getcwd(), 'model', 'bananacount.joblib')
banana_count_model = load(banana_count_model_path)
print(f"Banana count model loaded from {banana_count_model_path}")

print("Loading the health check model...")
health_check_model_path = os.path.join(os.getcwd(), 'model', '20m300x300.joblib')
health_check_model = load(health_check_model_path)
print(f"Health check model loaded from {health_check_model_path}")

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created upload folder at {UPLOAD_FOLDER}")
else:
    print(f"Upload folder already exists at {UPLOAD_FOLDER}")

# Function to preprocess images for banana count
def preprocess_image(image):
    print("Preprocessing image for banana count...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

# Function to extract features using HOG for banana count
def extract_features(image):
    print("Extracting features using HOG for banana count...")
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)
    features, _ = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                      cells_per_block=cells_per_block, block_norm='L2-Hys', visualize=True)
    print("Feature extraction complete.")
    return features

# Function to preprocess images for health check
def preprocess_health_check_image(image):
    print("Preprocessing image for health check...")
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv_img], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([hsv_img], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([hsv_img], [2], None, [256], [0, 256])
    hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    return hist

# Function to tile image into patches for banana count
def tile_image_banana_count(image, tile_size=(250, 300), overlap=(70, 70)):
    print("Tiling image into patches for banana count...")
    return tile_image_generic(image, tile_size, overlap)

# Function to tile image into patches for health check
def tile_image_health_check(image, tile_size=(300, 300), overlap=(0,0)):
    print("Tiling image into patches for health check...")
    return tile_image_generic(image, tile_size, overlap)

# Generic tiling function
def tile_image_generic(image, tile_size, overlap):
    tiles = []
    img_height, img_width = image.shape[:2]
    tile_height, tile_width = tile_size
    overlap_y, overlap_x = overlap

    if img_height < tile_height or img_width < tile_width:
        # Image is smaller than the tile size, process the entire image as a single tile
        print("Image is smaller than the tile size, processing the entire image as a single tile")
        tiles.append(image)
    else:
        # Image is larger than the tile size, tile the image into patches
        for y in range(0, img_height - tile_height + 1, tile_height - overlap_y):
            for x in range(0, img_width - tile_width + 1, tile_width - overlap_x):
                tile = image[y:y + tile_height, x:x + tile_width]
                if tile.shape[0] == tile_height and tile.shape[1] == tile_width:
                    tiles.append(tile)
    print(f"Total tiles created: {len(tiles)}")
    return tiles

@app.route('/')
def index():
    print("Rendering index.html")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    print("Upload route accessed")
    if 'file' not in request.files:
        print("No file part in request")
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        print("No file selected for uploading")
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Saving file to {filepath}")
        file.save(filepath)
        print(f"File saved to {filepath}")
        if 'bananaCount' in request.form:
            return redirect(url_for('bananacount', filename=filename))
        elif 'healthCheck' in request.form:
            return redirect(url_for('healthcheck', filename=filename))
    print("Redirecting back to the upload page")
    return redirect(request.url)

@app.route('/bananacount/<filename>', methods=['GET'])
def bananacount(filename):
    print(f"Processing image for banana count: {filename}")
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"Reading image from {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error reading image: {filename}")
        return redirect(url_for('index'))
    
    tiles = tile_image_banana_count(image)
    
    banana_count = 0
    for tile in tiles:
        print(f"Processing tile of size {tile.shape} for banana count")
        preprocessed_tile = preprocess_image(tile)
        features = extract_features(preprocessed_tile)
        
        # Ensure the number of features matches what the model expects
        expected_features = 38880  # Update this with the actual number of features expected by the model
        if len(features) != expected_features:
            print(f"Feature mismatch for tile from image: {filename} (expected {expected_features}, got {len(features)})")
            continue
        
        features = np.array(features).reshape(1, -1)
        prediction = banana_count_model.predict(features)
        print(f"Prediction for current tile: {'banana' if prediction == 0 else 'non-banana'}")
        if prediction == 0:
            banana_count += 1
    
    if banana_count == 0:
        print("No banana plants detected in the image.")
        min_bunch_weight = 0
        max_bunch_weight = 0
        min_net_weight = 0
        max_net_weight = 0
    else:
        print(f"Total banana plants detected: {banana_count}")
        min_bunch_weight = banana_count * 25
        max_bunch_weight = banana_count * 30
        min_net_weight = banana_count * 13
        max_net_weight = banana_count * 14
    
    return render_template('bananacount.html', filename=filename, banana_count=banana_count,
                           min_bunch_weight=min_bunch_weight, max_bunch_weight=max_bunch_weight,
                           min_net_weight=min_net_weight, max_net_weight=max_net_weight)


@app.route('/healthcheck/<filename>', methods=['GET'])
def healthcheck(filename):
    print(f"Processing image for health check: {filename}")
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"Reading image from {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error reading image: {filename}")
        return redirect(url_for('index'))
    
    tiles = tile_image_health_check(image)
    
    sigatoka_count = 0
    fusarium_count = 0
    for tile in tiles:
        print(f"Processing tile of size {tile.shape} for health check")
        hist = preprocess_health_check_image(tile)
        
        hist = np.array(hist).reshape(1, -1)
        prediction = health_check_model.predict(hist)
        
        if prediction == 1:
            sigatoka_count += 1
        elif prediction == 2:
            fusarium_count += 1

    if sigatoka_count == 0 and fusarium_count == 0:
        diagnosis = "Healthy"
    elif sigatoka_count > 0 and fusarium_count > 0 and abs(sigatoka_count - fusarium_count) < 15:
        diagnosis = "Both Black Sigatoka and Fusarium Wilt"
    elif fusarium_count > sigatoka_count:
        diagnosis = "Fusarium Wilt"
    else:
        diagnosis = "Black Sigatoka"

    print(f"Diagnosis: {diagnosis}")
    
    return render_template('healthcheck.html', filename=filename, diagnosis=diagnosis, sigatoka_count=sigatoka_count, fusarium_count=fusarium_count)

@app.route('/images/<filename>')
def uploaded_file(filename):
    print(f"Serving image: {filename}")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/logo/<filename>')
def logo_file(filename):
    print(f"Serving logo image: {filename}")
    return send_from_directory(os.path.join(os.getcwd(), 'logo'), filename)

@app.route('/usermanual')
def usermanual():
    return render_template('usermanual.html')


if __name__ == '__main__':
    print("Starting Flask app")
    app.run(debug=True)
