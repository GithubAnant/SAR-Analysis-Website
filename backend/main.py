from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import numpy as np
import base64
import io
import os
from typing import Optional, List

from api.sentinel_api import sentinel_api
from processing.image_loader import ImageLoader
from processing.change_detection import ChangeDetector
from processing.vizualization import visualizer
from utils.config import config

app = FastAPI(
    title="SAR Analysis API",
    description="Synthetic Aperture Radar (SAR) image analysis for change detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_loader = ImageLoader()
change_detector = ChangeDetector()

# Path to assets folder
ASSETS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "SAR Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "hotspots": "/hotspots", 
            "analyze": "/analyze",
            "upload_analyze": "/upload-analyze",
            "test_images": "/test-images"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/hotspots")
async def get_hotspots():
    """Get available analysis hotspots"""
    return {"hotspots": config.HOTSPOTS}

@app.get("/test-images")
async def test_local_images():
    """Test local image assets availability"""
    try:
        missing_images = []
        available_images = []
        
        for location, hotspot in config.HOTSPOTS.items():
            before_path = os.path.join(ASSETS_PATH, hotspot.get("image_before", ""))
            after_path = os.path.join(ASSETS_PATH, hotspot.get("image_after", ""))
            
            if not os.path.exists(before_path):
                missing_images.append(f"{location}: {hotspot.get('image_before', 'N/A')}")
            else:
                available_images.append(f"{location}: {hotspot.get('image_before', 'N/A')}")
                
            if not os.path.exists(after_path):
                missing_images.append(f"{location}: {hotspot.get('image_after', 'N/A')}")
            else:
                available_images.append(f"{location}: {hotspot.get('image_after', 'N/A')}")
        
        return {
            "status": "success" if not missing_images else "warning",
            "assets_path": ASSETS_PATH,
            "available_images": available_images,
            "missing_images": missing_images,
            "total_locations": len(config.HOTSPOTS)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error checking images: {str(e)}"
        }

@app.post("/analyze")
async def analyze_location(
    location: str = Form(...),
    date_before: str = Form(...),
    date_after: str = Form(...),
    bbox_size: float = Form(default=0.1)
):
    """
    Analyze changes at a specific location using Sentinel Hub data
    
    Args:
        location: Hotspot key (e.g., 'amazon', 'jakarta') or 'custom'
        date_before: Date in YYYY-MM-DD format
        date_after: Date in YYYY-MM-DD format  
        bbox_size: Bounding box size in degrees
    """
    try:
        # Validate dates
        try:
            date_before_obj = datetime.strptime(date_before, "%Y-%m-%d")
            date_after_obj = datetime.strptime(date_after, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        if date_after_obj <= date_before_obj:
            raise HTTPException(status_code=400, detail="date_after must be after date_before")
        
        # Get location and image files
        if location in config.HOTSPOTS:
            hotspot = config.HOTSPOTS[location]
            lat, lon = hotspot["lat"], hotspot["lon"]
            bbox_size = hotspot.get("bbox_size", bbox_size)
            image_before_filename = hotspot.get("image_before")
            image_after_filename = hotspot.get("image_after")
        else:
            raise HTTPException(status_code=400, detail=f"Unknown location: {location}")
        
        # Load images from assets folder
        print(f"Loading local images for {location}: {image_before_filename} and {image_after_filename}")
        
        if not image_before_filename or not image_after_filename:
            raise HTTPException(
                status_code=500,
                detail=f"Image files not configured for location: {location}"
            )
        
        image_before_path = os.path.join(ASSETS_PATH, image_before_filename)
        image_after_path = os.path.join(ASSETS_PATH, image_after_filename)
        
        # Check if image files exist
        if not os.path.exists(image_before_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Before image not found: {image_before_filename}"
            )
            
        if not os.path.exists(image_after_path):
            raise HTTPException(
                status_code=404, 
                detail=f"After image not found: {image_after_filename}"
            )
        
        # Load images from file paths
        image_before = image_loader.load_from_path(image_before_path)
        image_after = image_loader.load_from_path(image_after_path)
        
        if image_before is None or image_after is None:
            raise HTTPException(status_code=500, detail="Failed to process SAR images")
        
        # Detect changes
        results = change_detector.detect_changes(image_before, image_after)
        
        if not results:
            raise HTTPException(status_code=500, detail="Change detection failed")
        
        # Create visualizations
        side_by_side = visualizer.create_side_by_side(image_before, image_after)
        dashboard = visualizer.create_analysis_dashboard(
            image_before, image_after, results['change_mask'], results['statistics'], bbox_size
        )
        
        # Convert visualizations to base64
        response_data = {
            "location": location,
            "coordinates": {"lat": lat, "lon": lon},
            "date_before": date_before,
            "date_after": date_after,
            "images_used": {
                "before": image_before_filename,
                "after": image_after_filename
            },
            "statistics": results['statistics'],
            "visualizations": {
                "side_by_side": image_loader.array_to_base64(side_by_side) if side_by_side is not None else None,
                "dashboard": image_loader.array_to_base64(dashboard) if dashboard is not None else None
            }
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/upload-analyze")
async def upload_analyze(
    image_before: UploadFile = File(...),
    image_after: UploadFile = File(...)
):
    """
    Analyze changes between two uploaded images
    
    Args:
        image_before: Before image file
        image_after: After image file
    """
    try:
        # Validate file types
        allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/tiff"]
        
        if image_before.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Unsupported file type for before image")
        
        if image_after.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Unsupported file type for after image")
        
        # Read uploaded files
        before_bytes = await image_before.read()
        after_bytes = await image_after.read()
        
        # Load images
        img_before = image_loader.load_from_bytes(before_bytes)
        img_after = image_loader.load_from_bytes(after_bytes)
        
        if img_before is None or img_after is None:
            raise HTTPException(status_code=500, detail="Failed to process uploaded images")
        
        # Detect changes
        results = change_detector.detect_changes(img_before, img_after)
        
        if not results:
            raise HTTPException(status_code=500, detail="Change detection failed")
        
        # Create visualizations
        side_by_side = visualizer.create_side_by_side(img_before, img_after)
        dashboard = visualizer.create_analysis_dashboard(
            img_before, img_after, results['change_mask'], results['statistics']
        )
        
        # Prepare response
        response_data = {
            "analysis_type": "uploaded_images",
            "image_before_name": image_before.filename,
            "image_after_name": image_after.filename,
            "statistics": results['statistics'],
            "visualizations": {
                "side_by_side": image_loader.array_to_base64(side_by_side) if side_by_side is not None else None,
                "dashboard": image_loader.array_to_base64(dashboard) if dashboard is not None else None
            }
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload analysis failed: {str(e)}")

# Note: Using local images instead of Sentinel API

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)