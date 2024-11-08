from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from beavervision.api import router
from beavervision.utils.monitoring import init_monitoring
from beavervision.config import settings

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

# Static files setup
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Serve test page at /test
@app.get("/test")
async def test_page():
    return FileResponse(str(static_dir / "test.html"))

# Include API router
app.include_router(router, prefix=settings.API_V1_STR)