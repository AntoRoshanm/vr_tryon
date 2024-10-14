import os
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory, render_template_string, Response
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity
import mediapipe as mp  # For skeleton detection and face mesh detection

app = Flask(__name__)

# Load VGG16 model pre-trained on ImageNet and remove the top layers to use as a feature extractor
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)

# Function to load and preprocess image for VGG16
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image {image_path}")
        return None
    img = cv2.resize(img, (224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Extract features using VGG16
def extract_features(image_path):
    img_array = load_and_preprocess_image(image_path)
    if img_array is None:
        return None
    features = model.predict(img_array)
    return features.flatten()

# Function to load dataset dress images and extract features
def load_dress_collection(collection_folder):
    dress_collection = []
    dress_names = []
    for dress_file in os.listdir(collection_folder):
        if dress_file.endswith(('.jpg', '.jpeg', '.png')):  # Only consider image files
            image_path = os.path.join(collection_folder, dress_file)
            feature_vector = extract_features(image_path)
            if feature_vector is not None:
                dress_collection.append(feature_vector)
                dress_names.append(dress_file)
    return dress_collection, dress_names

# Load the dress collection from the 'dress_collection' folder
dress_collection, dress_names = load_dress_collection('dress_collection')

# Main page template
MAIN_PAGE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Home</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f0f0f0; display: flex; justify-content: center; align-items: center; height: 100vh; }
        .container { text-align: center; }
        button { padding: 15px 30px; font-size: 1.2rem; margin: 15px; border: none; border-radius: 8px; cursor: pointer; }
        .vr-button { background-color: #4CAF50; color: white; }
        .upload-button { background-color: #2196F3; color: white; }
        button i { margin-right: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Welcome to the Virtual Experience</h1>
        <button class="vr-button" onclick="location.href='/vr_try_on'"><i class="fas fa-vr-cardboard"></i> VR Try-On</button>
        <button class="upload-button" onclick="location.href='/upload_page'"><i class="fas fa-upload"></i> Upload Dress Image</button>
    </div>
    <script src="https://kit.fontawesome.com/a076d05399.js" crossorigin="anonymous"></script>
</body>
</html>
'''

# VR Try-On Page with left video and right dress grid
selected_dress = None  # Store the selected dress globally

@app.route('/select_dress', methods=['POST'])
def select_dress():
    global selected_dress
    data = request.get_json()
    selected_dress = data['dress']  # Store the selected dress
    return jsonify({'status': 'success', 'selected_dress': selected_dress})

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

VR_PAGE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VR Try-On</title>
    <style>
        body, html { margin: 0; padding: 0; height: 100%; display: flex; }

        /* Left side: Camera */
        #videoContainer {
            width: 50%;
            position: relative;
        }

        #videoElement {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        #overlayCanvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }

        /* Right side: Cards */
        #cardSection {
            width: 50%;
            padding: 20px;
            display: grid;
            grid-template-columns: repeat(2, 1fr);  /* 2 columns */
            grid-template-rows: repeat(5, 1fr);    /* 5 rows */
            gap: 10px;
        }

        .card {
            width: 100%;
            height: 100px;
            background-color: #f8f8f8;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 1.2rem;
            color: #444;
            cursor: pointer;
            transition: transform 0.3s ease;
        }

        .card img {
            max-width: 100%;
            max-height: 100%;
            border-radius: 8px;
        }

        .card:hover {
            transform: scale(1.05);
        }
    </style>
</head>
<body>

    <!-- Left side: Live Video with Skeleton Detection -->
    <div id="videoContainer">
        <img id="videoElement" src="{{ url_for('video_feed') }}" alt="Live video feed"/>
        <canvas id="overlayCanvas"></canvas>
    </div>

    <!-- Right side: Dress Selection Grid -->
    <div id="cardSection">
        {% for i in range(1, 11) %}  <!-- 5 rows, 2 columns = 10 dresses -->
        <div class="card" onclick="selectDress('{{i}}.png')">
            <img src="{{ url_for('tryon_image', filename=i|string+'.png') }}" alt="Dress {{i}}">
        </div>
        {% endfor %}
    </div>

    <script>
        function selectDress(imageName) {
            fetch('/select_dress', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ dress: imageName })
            }).then(response => response.json())
              .then(data => console.log('Selected dress:', data));
        }
    </script>

</body>
</html>
'''

# Function to overlay PNG with transparency, ensuring boundaries are respected
def overlayPNG(background, overlay, pos):
    h, w = overlay.shape[0], overlay.shape[1]
    x, y = pos[0], pos[1]

    # Ensure the overlay doesn't go beyond the background's boundaries
    if x >= background.shape[1] or y >= background.shape[0]:
        return background  # If out of bounds, return original background

    if x + w > background.shape[1]:
        w = background.shape[1] - x  # Adjust width to fit
        overlay = overlay[:, :w]  # Crop overlay to the available width

    if y + h > background.shape[0]:
        h = background.shape[0] - y  # Adjust height to fit
        overlay = overlay[:h, :]  # Crop overlay to the available height

    alpha_s = overlay[:, :, 3] / 255.0  # Alpha channel for transparency
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):  # Loop over color channels
        background[y:y+h, x:x+w, c] = (alpha_s * overlay[:, :, c] + alpha_l * background[y:y+h, x:x+w, c])

    return background

# Generator function for streaming video with skeleton detection and dress overlay
def gen_frames():
    global selected_dress
    cap = cv2.VideoCapture(0)  # Use webcam
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
    mp_drawing = mp.solutions.drawing_utils

    shirtFolderPath = "tryon"
    fixedRatio = 262 / 190  # widthOfShirt/widthOfPoint11to12
    shirtRatioHeightWidth = 581 / 440
    size_multiplier = 1.3  # Increase the dress size by a scaling factor

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with Pose detection (for shoulders)
        results_pose = pose.process(frame_rgb)

        # Process with Face Mesh to detect face points
        results_face = mp_face_mesh.process(frame_rgb)

        if results_pose.pose_landmarks and results_face.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lmList = results_pose.pose_landmarks.landmark

            # Get shoulder landmarks
            lm11 = lmList[11]  # Left shoulder
            lm12 = lmList[12]  # Right shoulder

            # Convert normalized coordinates to pixel values
            lm11 = (int(lm11.x * frame.shape[1]), int(lm11.y * frame.shape[0]))
            lm12 = (int(lm12.x * frame.shape[1]), int(lm12.y * frame.shape[0]))

            # Calculate shirt width and height using dynamic scaling logic
            shoulder_distance = abs(lm11[0] - lm12[0])
            if shoulder_distance > 0 and selected_dress:
                img_shirt = cv2.imread(os.path.join(shirtFolderPath, selected_dress), cv2.IMREAD_UNCHANGED)

                # Increase the size multiplier dynamically based on shoulder size
                width_of_shirt = int(shoulder_distance * fixedRatio * size_multiplier)
                height_of_shirt = int(width_of_shirt * shirtRatioHeightWidth)

                # Resize the shirt image
                img_shirt = cv2.resize(img_shirt, (width_of_shirt, height_of_shirt))

                # Calculate offset based on current scaling
                current_scale = shoulder_distance / 190
                offset = (int(44 * current_scale), int(48 * current_scale))

                # Position the shirt based on the left shoulder position
                position_x = lm12[0] - offset[0]
                position_y = lm12[1] - offset[1]

                # Overlay the shirt on the video frame
                try:
                    frame = overlayPNG(frame, img_shirt, (position_x, position_y))
                except Exception as e:
                    print(f"Error overlaying shirt: {e}")

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame to be displayed in the video stream
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# HTML template embedded in Python for the upload page
UPLOAD_PAGE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dress Search</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Arial', sans-serif; background: linear-gradient(135deg, #f8fafc, #dde1e7); min-height: 100vh; display: flex; justify-content: center; align-items: center; color: #333; }
        h1 { text-align: center; font-size: 2.5rem; color: #444; margin-bottom: 20px; }
        #uploadForm { background: #fff; padding: 20px 40px; border-radius: 10px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1); width: 100%; max-width: 400px; text-align: center; }
        #image-section { display: flex; justify-content: space-between; margin-top: 20px; }
        .image-container { width: 45%; text-align: center; }
        img { max-width: 100%; border-radius: 10px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1); }
        input[type="file"] { display: none; }
        label { background-color: #4A90E2; color: white; padding: 12px 20px; font-size: 1rem; border-radius: 25px; cursor: pointer; margin-bottom: 20px; display: inline-block; transition: background 0.3s ease; }
        label:hover { background-color: #357ABD; }
        button { background-color: #5cb85c; color: white; border: none; padding: 12px 25px; font-size: 1rem; border-radius: 25px; cursor: pointer; margin-top: 10px; transition: background 0.3s ease; }
        button:hover { background-color: #4cae4c; }
        #result { margin-top: 20px; font-size: 1.1rem; word-wrap: break-word; }
    </style>
</head>
<body>
    <div id="container">
        <h1>Upload Dress Image</h1>
        <form id="uploadForm" enctype="multipart/form-data">
            <label for="imageUpload">Choose a dress image</label>
            <input type="file" id="imageUpload" name="image" accept="image/*" required>
            <button type="submit">Search</button>
        </form>
        <div id="image-section">
            <div class="image-container">
                <h3>Your Uploaded Image</h3>
                <img id="userImage" src="" alt="No Image Uploaded">
            </div>
            <div class="image-container">
                <h3>Matched Image</h3>
                <img id="matchedImage" src="" alt="No Match Found">
            </div>
        </div>
        <div id="result"></div>
    </div>
    <script>
        document.getElementById('uploadForm').onsubmit = async (event) => {
            event.preventDefault();
            const formData = new FormData();
            const file = document.getElementById('imageUpload').files[0];
            formData.append('image', file);

            // Show the uploaded image on the left side
            const reader = new FileReader();
            reader.onload = function(e) {
                document.getElementById('userImage').src = e.target.result;
            };
            reader.readAsDataURL(file);

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
            });

            const result = await response.json();
            document.getElementById('result').innerHTML = result.message;

            // If a match is found, show the matched image on the right side
            if (result.matched_image) {
                document.getElementById('matchedImage').src = result.matched_image;
            } else {
                document.getElementById('matchedImage').src = '';
                document.getElementById('matchedImage').alt = 'No Match Found';
            }
        };
    </script>
</body>
</html>
'''

@app.route('/')
def main_page():
    return render_template_string(MAIN_PAGE_TEMPLATE)

@app.route('/upload_page')
def upload_page():
    return render_template_string(UPLOAD_PAGE_TEMPLATE)

@app.route('/vr_try_on')
def vr_try_on():
    return render_template_string(VR_PAGE_TEMPLATE)

@app.route('/tryon/<filename>')
def tryon_image(filename):
    return send_from_directory('tryon', filename)

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    if not file:
        return jsonify({'message': 'No file uploaded'})

    # Save and process the uploaded image
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Extract features from uploaded image
    uploaded_image_vector = extract_features(file_path)
    if uploaded_image_vector is None:
        return jsonify({'message': 'Error processing uploaded image.'})

    # Compare with dress collection using cosine similarity
    best_match_index = -1
    best_similarity_score = 0

    for idx, dress_image_vector in enumerate(dress_collection):
        similarity = cosine_similarity([uploaded_image_vector], [dress_image_vector])[0][0]
        if similarity > best_similarity_score:
            best_similarity_score = similarity
            best_match_index = idx

    if best_match_index == -1 or best_similarity_score < 0.5:  # Adjusted threshold to 0.5
        return jsonify({'message': 'Sorry, I do not have this dress.', 'matched_image': None})
    else:
        matched_image = dress_names[best_match_index]
        matched_image_url = f'/dress_collection/{matched_image}'  # URL to serve the image
        return jsonify({
            'message': f'Matching dress found: {matched_image} with similarity score: {best_similarity_score:.2f}',
            'matched_image': matched_image_url
        })

@app.route('/dress_collection/<filename>')
def send_image(filename):
    return send_from_directory('dress_collection', filename)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('tryon'):
        os.makedirs('tryon')  # Ensure the tryon directory exists
    app.run(debug=True)