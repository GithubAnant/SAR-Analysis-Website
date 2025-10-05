# ReFlecta - SAR Analysis Platform

**ReFlecta** is a real-time satellite image change detection and analysis platform that uses computer vision to identify and visualize environmental changes over time. The system analyzes before-and-after images from various global locations including Antarctica glaciers, deforestation areas, oil spills, ports, and military installations.

## Architecture Overview

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Web Interface<br/>HTML/CSS/JS]
        TABS[Tab Navigation<br/>Hotspot | Upload]
    end
    
    subgraph "API Gateway"
        API[FastAPI Server<br/>Port: 8000]
    end
    
    subgraph "Core Processing Engine"
        IMG[Image Loader<br/>Local Assets]
        CD[Change Detection<br/>Advanced Algorithms]
        VIZ[Visualization Engine<br/>Dashboard Generator]
    end
    
    subgraph "Data Sources"
        ASSETS[Local Image Assets<br/>antarctica1.webp, iraq1.webp, etc.]
        UPLOAD[User Uploads<br/>Before/After Images]
    end
    
    subgraph "Analysis Pipeline"
        PREP[Image Preprocessing<br/>Resize, Normalize]
        DETECT[Multi-Algorithm Detection<br/>• Structural Similarity<br/>• Perceptual Color Change<br/>• Gaussian Mixture Model<br/>• Consensus Voting]
        POST[Post Processing<br/>Noise Reduction, Filtering]
        STATS[Statistics Generation<br/>Change %, Regions, Pixels]
    end
    
    subgraph "Output Generation"
        SIDEBYSIDE[Side-by-Side View]
        DASHBOARD[Analysis Dashboard]
        JSON[Statistics JSON]
    end
    
    UI --> API
    TABS --> API
    
    API --> IMG
    API --> CD
    API --> VIZ
    
    IMG --> ASSETS
    IMG --> UPLOAD
    
    CD --> PREP
    PREP --> DETECT
    DETECT --> POST
    POST --> STATS
    
    VIZ --> SIDEBYSIDE
    VIZ --> DASHBOARD
    STATS --> JSON
    
    SIDEBYSIDE --> UI
    DASHBOARD --> UI
    JSON --> UI
    
    style UI fill:#e1f5fe
    style API fill:#f3e5f5
    style CD fill:#e8f5e8
    style DETECT fill:#fff3e0
    style ASSETS fill:#fce4ec
```

### Component Details

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | HTML5, CSS3, JavaScript, Tailwind | User interface and interaction |
| **API Gateway** | FastAPI, Python | RESTful API endpoints and request handling |
| **Image Loader** | OpenCV, PIL | Load and preprocess images from local assets |
| **Change Detection** | OpenCV, NumPy, SciPy | Multi-algorithm change detection pipeline |
| **Visualization** | Matplotlib, OpenCV | Generate comparison views and dashboards |
| **Data Storage** | Local File System | Store image assets and temporary uploads |

### Processing Pipeline

1. **Image Acquisition**: Load before/after images from local assets or user uploads
2. **Preprocessing**: Resize, normalize, and prepare images for analysis
3. **Multi-Algorithm Detection**:
   - Structural Similarity Index (SSIM)
   - Perceptual Color Change Detection
   - Gaussian Mixture Model (GMM)
   - Consensus voting across algorithms
4. **Post-Processing**: Filter noise, remove small artifacts, validate regions
5. **Visualization**: Generate side-by-side comparisons and analysis dashboards
6. **Statistics**: Calculate change percentages, affected areas, and region counts


## How to Run the Application

### Prerequisites
- Python 3.8 or higher

### Step 1: Clone the Repository
```bash
git clone https://github.com/GithubAnant/SAR-Analysis-Website.git
cd SAR-Analysis-Website
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

### Step 4: Verify Assets Directory
Ensure the `assets/` directory contains the required image files:
```bash
ls assets/
# Should show: antarctica1.webp, antarctica2.webp, iraq1.webp, iraq2.webp, etc.
```

### Step 5: Start the Backend Server
```bash
python main.py
```
The backend API will be available at: `http://localhost:8000`

### Step 6: Start the Frontend
Open a new terminal and navigate to the project root:
```bash
# From the project root directory
python -m http.server 3000
```
The frontend will be available at: `http://localhost:3000`

### Step 7: Access the Application
1. Open your web browser
2. Go to `http://localhost:3000`
3. Use the ReFlecta interface to:
   - Select from 10 predefined global hotspots (Antarctica, Iraq, Rotterdam Port, etc.)
   - Upload your own before/after images for custom analysis
   - View detailed change detection results with statistics and visualizations

## API Endpoints
- `GET /` - API information and available endpoints
- `GET /health` - Health check and system status
- `GET /hotspots` - Available predefined analysis locations
- `POST /analyze` - Analyze changes for selected hotspot location
- `POST /upload-analyze` - Analyze user-uploaded before/after images
- `GET /test-images` - Test endpoint to verify image loading
- `GET /docs` - Interactive API documentation (Swagger UI)

## Features
- **Multi-Algorithm Change Detection**: Uses SSIM, perceptual color analysis, and GMM for accurate detection
- **Predefined Hotspots**: Analyze 10+ global locations (Antarctica, Iraq, Rotterdam Port, etc.)
- **Smart Filtering**: Ignores minor variations and focuses on significant changes
- **Advanced Visualization**: Side-by-side comparisons and comprehensive analysis dashboards
- **Upload Support**: Analyze your own before/after images with the same algorithms
- **Detailed Statistics**: Change percentages, affected regions, and pixel-level analysis
- **Noise Reduction**: Advanced post-processing to eliminate false positives
- **Consensus Voting**: Multiple algorithms vote on changes for improved accuracy

## Technologies Used
- **Backend**: FastAPI, Python 3.8+, OpenCV, NumPy, SciPy, Matplotlib, Pillow
- **Frontend**: HTML5, JavaScript (ES6+), Tailwind CSS, Custom CSS
- **Computer Vision**: OpenCV for image processing, SSIM for structural analysis
- **Data Source**: Local image assets, user uploads (JPEG, PNG, WEBP, TIFF)
- **API Documentation**: Swagger UI (FastAPI auto-generated)
- **Image Processing**: Multi-algorithm pipeline with consensus voting



## 🧠 Tech Stack

### Frontend
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-323330?style=for-the-badge&logo=javascript&logoColor=F7DF1E)

### Backend
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Node.js](https://img.shields.io/badge/Node.js-43853D?style=for-the-badge&logo=node.js&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-4B8BBE?style=for-the-badge&logo=python&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Sentinel Hub API](https://img.shields.io/badge/Sentinel_Hub_API-2C5BB4?style=for-the-badge&logo=esa&logoColor=white)

### Deployment
![GitHub Pages](https://img.shields.io/badge/GitHub_Pages-181717?style=for-the-badge&logo=github&logoColor=white)
![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)

### Development & Tools
![Visual Studio Code](https://img.shields.io/badge/VS%20Code-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white)
