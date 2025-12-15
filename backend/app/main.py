"""
FastAPI backend for Brain Tumor Detection System
"""
import io
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .inference import load_model, process_image_full, predict_image
from .config import get_model_path, CLASS_NAMES, DEFAULT_CONFIDENCE
from .schemas import (
    DetectionResponse,
    Detection,
    HealthResponse,
    BatchProcessingResponse,
    ImageDetectionResult
)

# Create FastAPI app
app = FastAPI(
    title="Brain Tumor Detection API",
    description="YOLOv10 with CBAM for brain tumor detection in MRI images",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load the model when the server starts"""
    try:
        load_model()
        print("✅ Model loaded successfully on startup")
    except Exception as e:
        print(f"❌ Failed to load model on startup: {e}")
        raise


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint - confirms server is running and model is loaded
    """
    try:
        model_path = get_model_path()
        model = load_model()
        
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_path=str(model_path),
            class_names=CLASS_NAMES
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_path=str(e),
            class_names=CLASS_NAMES
        )


@app.post("/api/detect")
async def detect_single_image(
    file: UploadFile = File(...),
    confidence: float = DEFAULT_CONFIDENCE
):
    """
    Detect brain tumors in a single image.
    Returns both the annotated image and detection metadata.
    """
    try:
        # Read uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process image
        annotated_image, detections = process_image_full(image_bgr, confidence)
        
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        annotated_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        # Create response with image and metadata
        # We'll return the image as a streaming response with detection metadata in headers
        detection_data = {
            "detections": detections,
            "num_detections": len(detections),
            "image_shape": list(image_bgr.shape)
        }
        
        # Return image with metadata in custom header
        import json
        return StreamingResponse(
            img_byte_arr,
            media_type="image/jpeg",
            headers={
                "X-Detection-Data": json.dumps(detection_data),
                "Access-Control-Expose-Headers": "X-Detection-Data"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/api/detect-json", response_model=DetectionResponse)
async def detect_single_image_json(
    file: UploadFile = File(...),
    confidence: float = DEFAULT_CONFIDENCE
):
    """
    Detect brain tumors in a single image.
    Returns only detection metadata (no image).
    """
    try:
        # Read uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run inference only
        detections = predict_image(image_bgr, confidence)
        
        return DetectionResponse(
            detections=[Detection(**d) for d in detections],
            image_shape=list(image_bgr.shape),
            num_detections=len(detections)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/api/batch")
async def process_batch_images(
    files: List[UploadFile] = File(...),
    confidence: float = DEFAULT_CONFIDENCE
):
    """
    Process multiple images and return a ZIP file containing annotated images.
    Also returns detection summary as JSON.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    try:
        # Create in-memory ZIP file
        zip_buffer = io.BytesIO()
        results = []
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in files:
                try:
                    # Read image
                    contents = await file.read()
                    nparr = np.frombuffer(contents, np.uint8)
                    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image_bgr is None:
                        results.append(ImageDetectionResult(
                            filename=file.filename,
                            detections=[],
                            num_detections=0,
                            status="error",
                            error="Invalid image file"
                        ))
                        continue
                    
                    # Process image
                    annotated_image, detections = process_image_full(image_bgr, confidence)
                    
                    # Save annotated image to ZIP
                    img_byte_arr = io.BytesIO()
                    annotated_image.save(img_byte_arr, format='JPEG', quality=95)
                    
                    # Add to ZIP with processed_ prefix
                    base_name = Path(file.filename).stem
                    zip_name = f"processed_{base_name}.jpg"
                    zipf.writestr(zip_name, img_byte_arr.getvalue())
                    
                    # Store result
                    results.append(ImageDetectionResult(
                        filename=file.filename,
                        detections=[Detection(**d) for d in detections],
                        num_detections=len(detections),
                        status="success"
                    ))
                    
                except Exception as e:
                    results.append(ImageDetectionResult(
                        filename=file.filename,
                        detections=[],
                        num_detections=0,
                        status="error",
                        error=str(e)
                    ))
        
        # Add summary JSON to ZIP
        zip_buffer.seek(0)
        
        # Create summary
        summary = BatchProcessingResponse(
            total_images=len(files),
            successful=len([r for r in results if r.status == "success"]),
            failed=len([r for r in results if r.status == "error"]),
            results=results
        )
        
        # Add summary as JSON file to ZIP
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr(
                "detection_summary.json",
                summary.model_dump_json(indent=2)
            )
        
        zip_buffer.seek(0)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"brain_tumor_detection_results_{timestamp}.zip"
        
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")


@app.post("/api/batch-json", response_model=BatchProcessingResponse)
async def process_batch_images_json(
    files: List[UploadFile] = File(...),
    confidence: float = DEFAULT_CONFIDENCE
):
    """
    Process multiple images and return only detection metadata (no ZIP).
    Useful for preview before downloading.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    try:
        results = []
        
        for file in files:
            try:
                # Read image
                contents = await file.read()
                nparr = np.frombuffer(contents, np.uint8)
                image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image_bgr is None:
                    results.append(ImageDetectionResult(
                        filename=file.filename,
                        detections=[],
                        num_detections=0,
                        status="error",
                        error="Invalid image file"
                    ))
                    continue
                
                # Run inference
                detections = predict_image(image_bgr, confidence)
                
                results.append(ImageDetectionResult(
                    filename=file.filename,
                    detections=[Detection(**d) for d in detections],
                    num_detections=len(detections),
                    status="success"
                ))
                
            except Exception as e:
                results.append(ImageDetectionResult(
                    filename=file.filename,
                    detections=[],
                    num_detections=0,
                    status="error",
                    error=str(e)
                ))
        
        return BatchProcessingResponse(
            total_images=len(files),
            successful=len([r for r in results if r.status == "success"]),
            failed=len([r for r in results if r.status == "error"]),
            results=results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Brain Tumor Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "detect_single": "/api/detect",
            "detect_single_json": "/api/detect-json",
            "batch_process": "/api/batch",
            "batch_process_json": "/api/batch-json"
        },
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

