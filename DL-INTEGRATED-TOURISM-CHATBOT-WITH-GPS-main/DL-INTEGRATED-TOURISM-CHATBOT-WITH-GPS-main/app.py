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
model = tf.keras.models.load_model('my_model.keras')

# Load intents file with utf-8 encoding
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

# Load words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define function to process user input and generate responses
def get_response(intent_name):
    for intent in intents['intents']:
        if intent['tag'] == intent_name:
            return random.choice(intent['responses'])

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

@app.route('/')
def index():
    return render_template('index.html')  # Assuming your main HTML file is named index.html

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    predicted_intent = predict_class(message)
    response = get_response(predicted_intent)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
