from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import torch
from pathlib import Path
import logging

from api import router
from utils.monitoring import init_monitoring
from utils.logger import setup_logger
from config.settings import settings

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

# Debug middleware to log all requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

# Get the directory containing main.py
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Ensure static directory exists
STATIC_DIR.mkdir(exist_ok=True)

# Save the HTML files if they don't exist
if not (STATIC_DIR / "index.html").exists():
    logger.info("Creating index.html")
    with open(STATIC_DIR / "index.html", "w") as f:
        f.write("""[Your index.html content here]""")

if not (STATIC_DIR / "test.html").exists():
    logger.info("Creating test.html")
    with open(STATIC_DIR / "test.html", "w") as f:
        f.write("""[Your test.html content here]""")

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def root():
    """Serve the main page."""
    logger.info(f"Serving index.html from {STATIC_DIR / 'index.html'}")
    if not (STATIC_DIR / "index.html").exists():
        logger.error("index.html not found")
        return JSONResponse(
            status_code=404,
            content={"detail": "index.html not found"}
        )
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/test")
async def test_page():
    """Serve the test page."""
    logger.info(f"Serving test.html from {STATIC_DIR / 'test.html'}")
    if not (STATIC_DIR / "test.html").exists():
        logger.error("test.html not found")
        return JSONResponse(
            status_code=404,
            content={"detail": "test.html not found"}
        )
    return FileResponse(str(STATIC_DIR / "test.html"))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "static_dir_exists": STATIC_DIR.exists(),
        "index_html_exists": (STATIC_DIR / "index.html").exists(),
        "test_html_exists": (STATIC_DIR / "test.html").exists()
    }

# Include API router
app.include_router(router, prefix=settings.API_V1_STR)

# Print all registered routes at startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting BeaverVision API")
    logger.info(f"Static directory: {STATIC_DIR}")
    logger.info("Registered routes:")
    for route in app.routes:
        logger.info(f"Route: {route.path} [{', '.join(route.methods)}]")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)