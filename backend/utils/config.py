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
    
    # Processing parameters - Balanced sensitivity
    CHANGE_THRESHOLD = 25.0  # Moderate threshold for balanced detection
    MIN_CHANGE_AREA = 150    # Smaller minimum area for better detection
    COLOR_CHANGE_THRESHOLD = 30.0  # Lower Delta E threshold for more sensitivity
    CONTRAST_THRESHOLD = 8.0  # Lower contrast requirement
    
    # Hotspot coordinates and local image mappings
    HOTSPOTS = {
        "antarctica_glaciers": {
            "name": "Antarctica Glaciers",
            "lat": -77.8419,
            "lon": 166.6863,
            "bbox_size": 0.1,
            "image_before": "antarctica1.webp",
            "image_after": "antarctica2.webp"
        },
        "iraq": {
            "name": "Iraq Military Base", 
            "lat": 33.3152,
            "lon": 44.3661,
            "bbox_size": 0.1,
            "image_before": "iraq1.webp",
            "image_after": "iraq2.webp"
        },
        "rotterdam_port": {
            "name": "Rotterdam Port",
            "lat": 51.9225,
            "lon": 4.4792,
            "bbox_size": 0.1,
            "image_before": "rotterdam1.png",
            "image_after": "rotterdam2.png"
        },
        "singapore_port": {
            "name": "Singapore Port",
            "lat": 1.2966,
            "lon": 103.8764,
            "bbox_size": 0.1,
            "image_before": "port1.webp",
            "image_after": "port2.webp"
        },
        "amazon_rainforest": {
            "name": "Amazon Rainforest",
            "lat": -3.4653,
            "lon": -62.2159,
            "bbox_size": 0.1,
            "image_before": "deforestation1.png",
            "image_after": "deforestation2.png"
        },
        "oman_oil_spill": {
            "name": "Oman Oil Spill",
            "lat": 20.5937,
            "lon": 56.8974,
            "bbox_size": 0.1,
            "image_before": "oil1.webp",
            "image_after": "oil2.webp"
        },
        "military_base": {
            "name": "Military Base",
            "lat": 35.0000,
            "lon": 45.0000,
            "bbox_size": 0.1,
            "image_before": "site1.webp",
            "image_after": "site2.webp"
        },
        "deserted_border": {
            "name": "Desert Border",
            "lat": 30.0000,
            "lon": 35.0000,
            "bbox_size": 0.1,
            "image_before": "border1.webp",
            "image_after": "border2.webp"
        },
        "european_port": {
            "name": "European Port",
            "lat": 52.3676,
            "lon": 4.9041,
            "bbox_size": 0.1,
            "image_before": "europe1.png",
            "image_after": "europe2.png"
        },
        "gulf_of_bothnia": {
            "name": "Gulf of Bothnia Ice Cap",
            "lat": 63.8467,
            "lon": 20.2490,
            "bbox_size": 0.1,
            "image_before": "ice1.webp",
            "image_after": "ice2.webp"
        }
    }

config = Config()