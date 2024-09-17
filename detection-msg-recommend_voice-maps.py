import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys
import time
import pywhatkit
import threading
from playsound import playsound
import requests
from gtts import gTTS
import webbrowser  # New import

# Ensure UTF-8 compatibility
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load Haarcascades for face and eye detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load your pre-trained model
model_path = r'C:\Users\supra\Downloads\Driver distraction system 7.04pm\eye_status_model.h5'
new_model = load_model(model_path)
new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Function to get nearby restaurants using Overpass API (OSM)
def get_nearby_restaurants_osm(latitude, longitude, radius=3000):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      node["amenity"="restaurant"](around:{radius},{latitude},{longitude});
      node["amenity"="cafe"](around:{radius},{latitude},{longitude});
    );
    out body;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    
    restaurants = [
        {
            "name": element["tags"].get("name", "Unnamed Restaurant"),
            "distance": radius,  # Placeholder for distance
            "lat": element["lat"],  # Latitude of restaurant
            "lon": element["lon"]   # Longitude of restaurant
        }
        for element in data['elements']
    ]
    return restaurants

# Function to select the nearest restaurant
def select_nearest_restaurant(restaurants):
    if not restaurants:
        return None
    return restaurants[0] if restaurants else None

# Function to generate a voice alert for the nearest restaurant
def generate_voice_alert(restaurant_name, distance, lat, lon):
    try:
        text = f"The nearest restaurant is {restaurant_name}, which is approximately {distance} meters ahead. You can grab a coffee and take a rest."
        tts = gTTS(text=text, lang='en')
        tts.save("alert.mp3")
        playsound("alert.mp3")

        # Redirect to Google Maps
        maps_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
        webbrowser.open(maps_url)  # Opens the location in Google Maps
        print(f"Opening Google Maps for location: {lat}, {lon}")

    except Exception as e:
        print(f"Voice alert error: {e}")

# Function to recommend a nearby restaurant
def recommend_restaurant(latitude, longitude):
    # Fetch nearby restaurants using OSM (Overpass API)
    restaurants = get_nearby_restaurants_osm(latitude, longitude)
    
    # Select the nearest restaurant
    nearest_restaurant = select_nearest_restaurant(restaurants)
    
    if nearest_restaurant:
        name = nearest_restaurant['name']
        distance = nearest_restaurant['distance']
        lat = nearest_restaurant['lat']
        lon = nearest_restaurant['lon']
        
        # Generate voice alert and open location in Google Maps
        generate_voice_alert(name, distance, lat, lon)
    else:
        print("No nearby restaurants found.")

# Function to send a distress message using pywhatkit
def send_distress_message():
    try:
        message = "Alert! The driver is feeling sleepy. Please pay attention!"
        mobile_number = "+918447888238"  # Replace with your mobile number
        pywhatkit.sendwhatmsg_instantly(mobile_number, message)
        print(f"WhatsApp message sent to {mobile_number}")
    except Exception as e:
        print(f"WhatsApp error: {e}")

# Function to handle simultaneous alerts: WhatsApp and Google Maps
def handle_alerts(latitude, longitude):
    # Run restaurant recommendation and WhatsApp alert concurrently
    threading.Thread(target=recommend_restaurant, args=(latitude, longitude)).start()  # Recommend restaurant
    threading.Thread(target=send_distress_message).start()  # Send WhatsApp message

# Try different camera indices to find the working one
cap = None
for i in range(5):  # Test the first 5 camera indices
    temp_cap = cv2.VideoCapture(i)
    if temp_cap.isOpened():
        cap = temp_cap
        break

if cap is None or not cap.isOpened():
    raise IOError("Cannot open webcam")

# Variables for tracking drowsiness
closed_eyes_start_time = None
alert_sent = False  # Track if the WhatsApp alert has been sent
restaurant_recommended = False  # Track if restaurant recommendation was given

# Coordinates (example: Bangalore, India)
latitude = 12.9716
longitude = 77.5946

# Main loop to process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if frame not read properly

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the frame
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

    status = "Closed Eyes"
    eyes_closed = True  # Assume eyes are closed

    for (x, y, w, h) in eyes:
        roi_color = frame[y:y + h, x:x + w]

        # Resize the eye region for the model input
        eyes_roi = cv2.resize(roi_color, (64, 64))
        eyes_roi = eyes_roi / 255.0  # Normalize the image
        final_image = np.expand_dims(eyes_roi, axis=0)  # Add batch dimension

        # Predict the status using the pre-trained model
        Predictions = new_model.predict(final_image)
        if Predictions[0][0] > 0.5:
            status = "Open Eyes"
            eyes_closed = False
            closed_eyes_start_time = None  # Reset the timer if eyes are opened
            restaurant_recommended = False  # Reset the recommendation flag
            alert_sent = False  # Reset the alert flag

    if eyes_closed:
        if closed_eyes_start_time is None:
            closed_eyes_start_time = time.time()  # Record the start time when eyes close
        elapsed_time = time.time() - closed_eyes_start_time

        # Trigger simultaneous actions after 8 seconds of continuous eye closure
        if elapsed_time >= 8 and not alert_sent:
            threading.Thread(target=handle_alerts, args=(latitude, longitude)).start()  # Handle both tasks in separate threads
            alert_sent = True  # Ensure alerts are sent only once
    else:
        closed_eyes_start_time = None  # Reset the timer if eyes are opened
        restaurant_recommended = False  # Reset the recommendation flag
        alert_sent = False  # Reset the alert flag

    # Display the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, status, (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Drowsiness Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()