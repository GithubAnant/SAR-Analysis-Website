# SAR Analysis Application

A comprehensive Synthetic Aperture Radar (SAR) analysis platform for real-time change detection using Sentinel-1 satellite imagery.


## How to Run the Application

### Prerequisites
- Python 3.8 or higher

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd SAR-analysis
```

### Step 2: Set up Virtual Environment (Recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a `.env` file in the root directory with your Sentinel Hub credentials:
```
SENTINEL_CLIENT_ID=your_client_id_here
SENTINEL_CLIENT_SECRET=your_client_secret_here
```

### Step 5: Start the Backend Server
```bash
python main.py
```
The backend API will be available at: `http://localhost:8000`

### Step 6: Start the Frontend
Open a new terminal and navigate to the frontend directory:
```bash
cd frontend
python -m http.server 3000
```
The frontend will be available at: `http://localhost:3000`

### Step 7: Access the Application
1. Open your web browser
2. Go to `http://localhost:3000`
3. Use the SAR Analysis interface to:
   - Analyze predefined hotspots with date ranges
   - Upload your own before/after images for analysis

## API Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `GET /hotspots` - Available analysis locations
- `POST /analyze` - Analyze hotspot changes
- `POST /upload-analyze` - Analyze uploaded images
- `GET /docs` - Interactive API documentation

## Features
- **Real-time SAR Analysis**: Connect to Sentinel Hub for live satellite data
- **Change Detection**: Advanced algorithms to detect surface changes
- **Visualization**: Multiple visualization modes including heatmaps and dashboards
- **Upload Support**: Analyze your own before/after images
- **Statistics**: Detailed change statistics and affected area calculations

## Technologies Used
- **Backend**: FastAPI, Python, OpenCV, NumPy, SciPy
- **Frontend**: HTML5, JavaScript, Tailwind CSS
- **Data Source**: Sentinel Hub API, Sentinel-1 SAR imagery