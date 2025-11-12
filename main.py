"""
FastAPI application for document vs selfie detection.

Endpoints:
    POST /detect - Detect if uploaded image is a document or selfie
    GET /health - Health check endpoint
"""
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.detector import DocumentDetector
from src.image_processor import ImageProcessor

# Initialize FastAPI app
app = FastAPI(
    title="Anti-Spoofing Document Detector",
    description="Service to detect document vs selfie images",
    version="1.0.0"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = DocumentDetector(use_ml=True)
image_processor = ImageProcessor()


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON with service status
    """
    return JSONResponse({
        "status": "healthy",
        "service": "Anti-Spoofing Document Detector",
        "version": "1.0.0"
    })


@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    """
    Detect if uploaded image is a document or selfie.
    
    Args:
        file: Uploaded image file (JPEG, PNG)
        
    Returns:
        JSON with detection result and optional confidence score
        
    Raises:
        HTTPException: If image is invalid or processing fails
    """
    try:
        # Read file contents
        contents = await file.read()
        
        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        # Validate and load image
        image = image_processor.validate_and_load(contents)
        
        # Perform detection
        result = detector.detect(image)
        
        return JSONResponse(result)
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/detect/batch")
async def detect_batch(files: list[UploadFile] = File(...)):
    """
    Process multiple images in batch.
    
    Args:
        files: List of image files
        
    Returns:
        JSON array with detection results for each image
        
    Raises:
        HTTPException: If processing fails
    """
    results = []
    
    try:
        for file in files:
            contents = await file.read()
            
            if not contents:
                results.append({
                    "filename": file.filename,
                    "error": "Empty file"
                })
                continue
            
            try:
                image = image_processor.validate_and_load(io.BytesIO(contents))
                detection = detector.detect(image)
                results.append({
                    "filename": file.filename,
                    **detection
                })
            except ValueError as e:
                results.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        return JSONResponse({"results": results})
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

