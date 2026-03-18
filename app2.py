from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import logging
import os
from typing import Dict, List
import uvicorn

app = FastAPI(title="YOLOv8 Traffic Density API")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
model = YOLO('yolov8n.pt')

# Vehicle classes in COCO dataset
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle', 
    5: 'bus',
    7: 'truck'
}

@app.get("/")
async def root():
    return {"message": "YOLOv8 Traffic Density API", "status": "running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "yolov8n",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/detect")
async def detect_vehicles(file: UploadFile = File(...)):
    """
    Upload an image for vehicle detection and traffic density estimation
    """
    try:
        # Read and process image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run detection
        results = model(img)
        
        # Process results
        vehicle_count = 0
        detections = []
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in VEHICLE_CLASSES:
                    vehicle_count += 1
                    detections.append({
                        'class': VEHICLE_CLASSES[cls],
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist()
                    })
        
        # Calculate traffic density (vehicles per 1000x1000 area)
        height, width = img.shape[:2]
        area_normalized = (height * width) / 1000000  # Normalize to megapixels
        density = vehicle_count / area_normalized if area_normalized > 0 else 0
        
        # Log to persistent storage if available
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'filename': file.filename,
            'vehicle_count': vehicle_count,
            'density': round(density, 2)
        }
        
        # Try to write to mounted volume
        try:
            with open('/mnt/logs/detections.log', 'a') as f:
                f.write(str(log_entry) + '\n')
        except:
            pass  # Volume not mounted
        
        return JSONResponse({
            'success': True,
            'filename': file.filename,
            'vehicle_count': vehicle_count,
            'traffic_density': round(density, 2),
            'detections': detections,
            'processing_time_ms': results[0].speed.get('inference', 0),
            'image_size': f"{width}x{height}"
        })
        
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """
    Get API metrics and statistics
    """
    # You can implement metrics collection here
    return {
        "uptime": "TODO",
        "total_requests": 0,
        "average_processing_time": 0
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)