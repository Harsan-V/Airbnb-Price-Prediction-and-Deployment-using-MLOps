import requests

url = "http://127.0.0.1:8000/predict"

payload = {
    "city": "Chennai",
    "latitude": 13.0542,
    "longitude": 80.2249,
    "rating": 4.94,
    "reviews": 119,
    "bedrooms": 1,
    "bathrooms": 1,
    "beds": 1,
    "guest_capacity": 2,
    "is_superhost": True,
    "host_rating": 4.94,
    "amenities_count": 24,
    "has_wifi": True,
    "has_kitchen": True,
    "has_ac": True,
    "has_parking": True,
    "has_washer": True,
    "has_tv": True,
    "has_pool": False,
    "has_workspace": True,
    "description_length": 450,
    "title_length": 22,
    "property_type_inferred": "Room"
}

response = requests.post(url, json=payload, timeout=10)
print("Status code:", response.status_code)
print("Response:", response.json())
