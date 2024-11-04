# main.py
from fastapi import FastAPI
from beavervision.api import router
from beavervision.utils.monitoring import init_monitoring
from beavervision.config import settings

# Initialize the FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for real-time lip synchronization with facial expressions",
    version="1.0.0"
)

# Initialize monitoring
init_monitoring()

# Include API router
app.include_router(router, prefix=settings.API_V1_STR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Enable auto-reload during development
    )