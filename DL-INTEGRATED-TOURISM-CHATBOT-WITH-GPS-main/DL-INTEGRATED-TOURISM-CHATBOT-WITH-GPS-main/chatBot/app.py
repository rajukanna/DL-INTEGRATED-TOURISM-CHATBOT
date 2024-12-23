from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import json
import random
import pickle
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__, static_folder='static')

# Load your trained model
model = tf.keras.models.load_model('static/my_model.keras')

# Load intents file with utf-8 encoding
with open('static/intents.json', encoding='utf-8') as file:
    intents = json.load(file)

# Load words and classes
words = pickle.load(open('static/words.pkl', 'rb'))
classes = pickle.load(open('static/classes.pkl', 'rb'))

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Dictionary containing latitude and longitude coordinates of Indian states and union territories
state_coordinates = {
    "Andhra Pradesh": (15.9129, 79.7400),
    "Arunachal Pradesh": (28.2180, 94.7278),
    "Assam": (26.2006, 92.9376),
    "Bihar": (25.0961, 85.3131),
    "Chhattisgarh": (21.2787, 81.8661),
    "Goa": (15.2993, 74.1240),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 76.0856),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139),
    "Manipur": (24.6637, 93.9063),
    "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1645, 92.9376),
    "Nagaland": (26.1584, 94.5624),
    "Odisha": (20.9517, 85.0985),
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Sikkim": (27.5330, 88.5122),
    "Tamil Nadu": (11.1271, 78.6569),
    "Telangana": (18.1124, 79.0193),
    "Tripura": (23.9408, 91.9882),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193),
    "West Bengal": (22.9868, 87.8550),
    "Andaman and Nicobar Islands": (11.7401, 92.6586),
    "Chandigarh": (30.7333, 76.7794),
    "Dadra and Nagar Haveli and Daman and Diu": (20.1809, 73.0169),
    "Lakshadweep": (10.5667, 72.6417),
    "Delhi": (28.7041, 77.1025),
    "Puducherry": (11.9416, 79.8083)
}

# Define function to process user input and generate responses
def get_response(intent_name):
    for intent in intents['intents']:
        if intent['tag'] == intent_name:
            return random.choice(intent['responses'])
    return "Sorry, I don't understand that."

def preprocess_input(text):
    # Tokenize and lemmatize the input text
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    
    # Create a bag-of-words array
    bag = [0] * len(words)
    for w in tokens:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
                
    return np.array([bag])

def predict_class(text):
    processed_text = preprocess_input(text)
    predictions = model.predict(processed_text)
    class_label = np.argmax(predictions)
    return classes[class_label]

def get_location_name(latitude, longitude):
    for state, (lat, lon) in state_coordinates.items():
        if latitude == lat and longitude == lon:
            return state
    return "Unknown location"

def calculate_distance(current_location, destination):
    current_lat, current_lon = current_location
    destination_lat, destination_lon = state_coordinates[destination]
    distance = ((current_lat - destination_lat) ** 2 + (current_lon - destination_lon) ** 2) ** 0.5
    return distance

@app.route('/')
def index():
    return render_template('index.html')  # Assuming your main HTML file is named index.html

# Flask route to handle chat requests
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data['message']
    print(f"Received message: {message}")  # Debug statement
    predicted_intent = predict_class(message)
    print(f"Predicted Intent: {predicted_intent}")  # Debug statement
    
    if predicted_intent == 'current_location':
        try:
            current_latitude = float(data['current_latitude'])
            current_longitude = float(data['current_longitude'])
            location_name = get_location_name(current_latitude, current_longitude)
            response = f"Your current location is {location_name}."
        except (KeyError, ValueError) as e:
            response = "Error retrieving location: Invalid or missing coordinates."
    elif predicted_intent == 'calculate_distance':
        try:
            destination = message.split("to ")[1].capitalize()  # Extract the destination from the message
            distance = calculate_distance((float(data['current_latitude']), float(data['current_longitude'])), destination)
            response = random.choice([resp.format(destination=destination, distance=distance) for resp in intents['calculate_distance']['responses']])
        except (IndexError, KeyError, ValueError) as e:
            response = "Error calculating distance: Invalid destination or missing coordinates."
    else:
        response = get_response(predicted_intent)
    
    print(f"Response: {response}")  # Debug statement
    
    return jsonify({'response': response, 'predicted_intent': predicted_intent})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)