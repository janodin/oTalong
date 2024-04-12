import base64
import cv2
import numpy as np
from django.shortcuts import render
from skimage.transform import resize

def load_model(filename):
    import pickle
    # Load the model from the pickle file
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

# Load the trained model
model = load_model('static/models.pkl')

# List of class names your model can predict
CATEGORIES = ['Healthy', 'Phomopsis']

def preprocess(image):
    # Resize and normalize the image
    processed_img = resize(image, (128, 128, 3)) / 255.0
    return processed_img

def extract_color_features(image):
    # Initialize the list to store color features
    color_features = []

    # Iterate through each color channel (R, G, B)
    for i in range(3):
        channel = image[:, :, i]
        # Compute and append statistical features of the channel
        features = [
            np.mean(channel),
            np.std(channel),
            np.median(channel),
            *np.percentile(channel, [25, 75])  # Add 25th and 75th percentiles
        ]
        color_features.extend(features)

    return np.array([color_features])

def predict_image(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')

        if image_file:
            img_array = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

            preprocessed_img = preprocess(img)
            image_features = extract_color_features(preprocessed_img)

            probabilities = model.predict_proba(image_features)[0]
            predicted_label_index = np.argmax(probabilities)
            predicted_label = CATEGORIES[predicted_label_index]
            accuracy = probabilities[predicted_label_index]

            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Convert back to BGR for encoding
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            image_data = f'data:image/jpeg;base64,{encoded_image}'
        else:
            image_data = 'No image provided'
            predicted_label = 'No prediction'
            accuracy = 'N/A'

        context = {
            'image_data': image_data,
            'predicted_label': predicted_label,
            'accuracy': f'{accuracy * 100:.2f}%' if accuracy != 'N/A' else accuracy
        }
        return render(request, 'output_image.html', context)

    return render(request, 'predict_image.html')
