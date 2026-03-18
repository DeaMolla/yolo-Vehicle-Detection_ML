import os
from fastapi import FastAPI
from ultralytics import YOLO

app = FastAPI()

# Load the model from your repo
model_path = "best.pt" if os.path.exists("best.pt") else "yolov8n.pt"
model = YOLO(model_path)

@app.get("/")
def read_root():
    return {"Project": "Real-Time Traffic Density Estimation", "Status": "Online"}

@app.get("/predict")
def predict():
    # This runs prediction on the sample video/image in your repo
    results = model.predict(source="sample_video.mp4", save=False, conf=0.5)
    
    # Count vehicle detections in the first result frame
    count = len(results[0].boxes)
    
    return {
        "vehicle_count": count,
        "density": "High" if count > 10 else "Low",
        "message": f"Detected {count} vehicles in the sample."
    }