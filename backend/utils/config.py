import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Sentinel Hub credentials
    SENTINEL_CLIENT_ID = os.getenv("SENTINEL_CLIENT_ID", "")
    SENTINEL_CLIENT_SECRET = os.getenv("SENTINEL_CLIENT_SECRET", "")
    
    # API endpoints
    SENTINEL_TOKEN_URL = "https://services.sentinel-hub.com/oauth/token"
    SENTINEL_PROCESS_URL = "https://services.sentinel-hub.com/api/v1/process"
    
    # Image settings
    IMAGE_WIDTH = 512
    IMAGE_HEIGHT = 512
    
    # Processing parameters
    CHANGE_THRESHOLD = 3.0  # dB threshold for change detection
    MIN_CHANGE_AREA = 100   # minimum pixels for valid change
    
    # Hotspot coordinates (lat, lon, name)
    HOTSPOTS = {
        "amazon": {
            "name": "Amazon Rainforest",
            "lat": -3.4653,
            "lon": -62.2159,
            "bbox_size": 0.1
        },
        "jakarta": {
            "name": "Jakarta, Indonesia",
            "lat": -6.2088,
            "lon": 106.8456,
            "bbox_size": 0.1
        },
        "iceland": {
            "name": "Iceland Volcano",
            "lat": 63.6318,
            "lon": -19.6083,
            "bbox_size": 0.1
        },
        "california": {
            "name": "California",
            "lat": 36.7783,
            "lon": -119.4179,
            "bbox_size": 0.1
        },
        "greenland": {
            "name": "Greenland Glacier",
            "lat": 69.2138,
            "lon": -49.9460,
            "bbox_size": 0.1
        }
    }

config = Config()