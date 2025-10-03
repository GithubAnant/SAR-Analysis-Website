from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import numpy as np
import base64
import io
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
            "upload_analyze": "/upload-analyze"
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

@app.get("/test-sentinel")
async def test_sentinel_connection():
    """Test Sentinel Hub API connection"""
    try:
        # Test authentication
        token = sentinel_api.get_access_token()
        if not token:
            return {
                "status": "error",
                "message": "Failed to get access token. Check your credentials.",
                "credentials_configured": bool(config.SENTINEL_CLIENT_ID and config.SENTINEL_CLIENT_SECRET)
            }
        
        return {
            "status": "success",
            "message": "Successfully connected to Sentinel Hub API",
            "credentials_configured": True,
            "token_length": len(token)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Connection failed: {str(e)}",
            "credentials_configured": bool(config.SENTINEL_CLIENT_ID and config.SENTINEL_CLIENT_SECRET)
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
        
        # Get location coordinates
        if location in config.HOTSPOTS:
            hotspot = config.HOTSPOTS[location]
            lat, lon = hotspot["lat"], hotspot["lon"]
            bbox_size = hotspot.get("bbox_size", bbox_size)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown location: {location}")
        
        # Fetch images from Sentinel Hub
        print(f"Fetching SAR images for {location} ({lat}, {lon})")
        
        # Check if Sentinel Hub credentials are configured
        if not config.SENTINEL_CLIENT_ID or not config.SENTINEL_CLIENT_SECRET:
            raise HTTPException(
                status_code=500,
                detail="Sentinel Hub credentials not configured. Please check your .env file."
            )
        
        image_before_bytes = await fetch_sar_image_async(lat, lon, date_before, bbox_size)
        if not image_before_bytes:
            raise HTTPException(
                status_code=404, 
                detail=f"Could not fetch SAR image for {date_before}. The image may not be available for this date/location."
            )
            
        image_after_bytes = await fetch_sar_image_async(lat, lon, date_after, bbox_size)
        if not image_after_bytes:
            raise HTTPException(
                status_code=404, 
                detail=f"Could not fetch SAR image for {date_after}. The image may not be available for this date/location."
            )
        
        # Load images
        image_before = image_loader.load_from_bytes(image_before_bytes)
        image_after = image_loader.load_from_bytes(image_after_bytes)
        
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

async def fetch_sar_image_async(lat: float, lon: float, date: str, bbox_size: float):
    """Async wrapper for SAR image fetching"""
    return sentinel_api.fetch_sar_image(lat, lon, date, bbox_size)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)