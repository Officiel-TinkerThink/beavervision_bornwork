from fastapi import FastAPI
from beavervision.api.endpoints import router
app = FastAPI(
    title="Lip Sync API",
    description="API for real-time lip synchronization with facial expressions",
    version="1.0.0"
)

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)