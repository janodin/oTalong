import base64
import cv2
import numpy as np
from django.shortcuts import render
from skimage.feature import local_binary_pattern
import pickle


def load_model_and_scaler():
    # Load the model and scaler from the pickle files
    with open('static/models.pkl', 'rb') as f:
        model_classifier = pickle.load(f)
    with open('static/scaler.pkl', 'rb') as f:
        scaler_classifier = pickle.load(f)
    return model_classifier, scaler_classifier


# Load the trained model and scaler
model, scaler = load_model_and_scaler()


def preprocess_and_extract_features(img):
    img = cv2.resize(img, (128, 128))  # Resize the image to match training data

    # Convert image to grayscale for LBP
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=24, R=3, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(27), range=(0, 26))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize histogram

    # Convert image to HSV for color histograms
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()

    # Concatenate all features
    feature_vector = np.hstack((lbp_hist, hist_h, hist_s, hist_v))
    return feature_vector


def predict_image(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if image_file:
            img_array = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            features = preprocess_and_extract_features(img)
            features_scaled = scaler.transform([features])

            probabilities = model.predict_proba(features_scaled)[0]  # Get probability estimates

            # Assuming class 0 is "Healthy", class 1 is "Phomopsis"
            healthy_prob = probabilities[0]
            infected_prob = probabilities[1]

            # Set a threshold for detection confidence
            threshold = 0.8  # Adjust based on validation results

            if max(healthy_prob, infected_prob) < threshold:
                predicted_label = 'No matching images found'
                accuracy = 'N/A'
            else:
                prediction = np.argmax(probabilities)
                predicted_label = "Healthy" if prediction == 0 else "Phomopsis"
                accuracy = max(healthy_prob, infected_prob)

            _, buffer = cv2.imencode('.jpg', img)
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
