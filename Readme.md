sar-analysis-app/
├── backend/
│   ├── main.py                 # FastAPI main server
│   ├── api/
│   │   ├── __init__.py
│   │   ├── sentinel_api.py     # Sentinel Hub API calls
│   │   └── earth_engine.py     # Google Earth Engine (alternative)
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── image_loader.py     # Load and parse SAR images
│   │   ├── change_detection.py # Detect changes between images
│   │   └── visualization.py    # Create visualizations
│   ├── utils/
│   │   ├── __init__.py
│   │   └── config.py           # API keys and config
│   └── requirements.txt
├── frontend/
│   └── index.html              # React app (I'll build this)
└── .env                        # API keys (gitignore this!)