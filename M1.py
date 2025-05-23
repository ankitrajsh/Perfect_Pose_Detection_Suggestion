import cv2
import requests
from datetime import datetime
from PIL import Image
from geopy.geocoders import Nominatim

# Optional: OpenWeatherMap API key
OPENWEATHER_API_KEY = "YOUR_API_KEY"  # Replace with your key

# ----- Step 1: Image Brightness Analysis -----
def get_lighting_condition(image_path):
    img = cv2.imread(image_path)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = grayscale.mean()

    if brightness < 50:
        return "Very Dim"
    elif brightness < 100:
        return "Dim"
    elif brightness < 180:
        return "Normal"
    else:
        return "Bright"

# ----- Step 2: Get Location Type from GPS -----
def get_location_type_from_gps(lat, lon):
    geolocator = Nominatim(user_agent="posepal")
    location = geolocator.reverse((lat, lon), language='en')
    return location.address if location else "Unknown location"

# ----- Step 3: Weather Detection -----
def get_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url).json()
    weather_desc = response['weather'][0]['description']
    return weather_desc

# ----- Step 4: Get Time of Day -----
def get_time_of_day():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 20:
        return "Evening"
    else:
        return "Night"

# ----- Step 5: Main Context Detection -----
def detect_context(image_path, gps_coords, event_type=None):
    lat, lon = gps_coords
    context = {
        "lighting_condition": get_lighting_condition(image_path),
        "location_description": get_location_type_from_gps(lat, lon),
        "weather": get_weather(lat, lon),
        "time_of_day": get_time_of_day(),
        "event_theme": event_type or "Not Specified"
    }
    return context

# ----- Run the module -----
if __name__ == "__main__":
    image_path = "Harini-Aswin-MCC-Hall-Chennai-0609+-+Copy.jpg"  # Replace with your image
    gps_coords = (19.0760, 72.8777)  # Example: Mumbai coordinates
    event_type = "Wedding"           # User input (optional)

    context = detect_context(image_path, gps_coords, event_type)
    print(" Extracted Context:")
    for k, v in context.items():
        print(f"  {k}: {v}")
