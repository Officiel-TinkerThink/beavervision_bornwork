from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRoute
import torch
from pathlib import Path
import logging

from beavervision.api import router
from beavervision.utils.monitoring import init_monitoring
from beavervision.config import settings
from beavervision.utils.logger import setup_logger

logger = setup_logger(__name__)

# Initialize the FastAPI application
app = FastAPI(
    title="BeaverVision API",
    description="Real-time lip synchronization API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory containing main.py
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Enhanced static directory initialization
def init_static_directory():
    """Initialize static directory and log its contents"""
    logger.info(f"Initializing static directory at: {STATIC_DIR}")
    
    # Create static directory if it doesn't exist
    STATIC_DIR.mkdir(exist_ok=True)
    
    # Log all files in static directory
    logger.info("Static directory contents:")
    for file_path in STATIC_DIR.glob("*"):
        logger.info(f"- {file_path.name}")
    
    return STATIC_DIR.exists()

# Enhanced debug middleware
@app.middleware("http")
async def debug_middleware(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    logger.info(f"Request headers: {request.headers}")
    logger.info(f"Static directory exists: {STATIC_DIR.exists()}")
    
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        if response.status_code == 404:
            logger.error(f"404 Not Found: {request.url.path}")
            logger.error(f"Attempted to access file: {STATIC_DIR / request.url.path.lstrip('/')}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

# Initialize static directory before mounting
if init_static_directory():
    # Mount static files with explicit static directory check
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    logger.error("Failed to initialize static directory")

@app.get("/test")
async def test_page():
    """Serve the test page with enhanced error handling"""
    test_file_path = STATIC_DIR / "test.html"
    logger.info(f"Attempting to serve test.html from: {test_file_path}")
    
    if not STATIC_DIR.exists():
        logger.error(f"Static directory does not exist: {STATIC_DIR}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Static directory not found"}
        )
    
    if not test_file_path.exists():
        logger.error(f"test.html not found at: {test_file_path}")
        return JSONResponse(
            status_code=404,
            content={"detail": f"test.html not found at {test_file_path}"}
        )
    
    logger.info("Successfully located test.html")
    return FileResponse(str(test_file_path))

# Enhanced health check endpoint
@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with detailed static file information"""
    static_files = list(STATIC_DIR.glob("*")) if STATIC_DIR.exists() else []
    
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "static_dir": {
            "path": str(STATIC_DIR),
            "exists": STATIC_DIR.exists(),
            "files": [str(f.name) for f in static_files]
        },
        "test_html": {
            "path": str(STATIC_DIR / "test.html"),
            "exists": (STATIC_DIR / "test.html").exists()
        }
    }