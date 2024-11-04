from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Lip Sync API"
    
    # Model Paths
    WAV2LIP_MODEL_PATH: str = "models/wav2lip_gan.pth"
    TTS_MODEL_PATH: str = "models/tts_model.pth"
    
    # Processing Settings
    MIN_VIDEO_DURATION: int = 5  # seconds
    MAX_VIDEO_DURATION: int = 30  # seconds
    MAX_VIDEO_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_VIDEO_TYPES: set = {"video/mp4", "video/avi"}
    
    # Device Settings
    CUDA_DEVICE: str = "cuda:0"
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Monitoring Settings
    ENABLE_METRICS: bool = True
    PROMETHEUS_PORT: int = 9090
    
    class Config:
        case_sensitive = True
        env_file = ".env"