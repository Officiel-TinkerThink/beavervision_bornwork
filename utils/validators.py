from fastapi import HTTPException
from moviepy.editor import VideoFileClip
import magic
from ..config.settings import Settings

def validate_video(file_path: str):
    """Validate video file format, size, and duration."""
    settings = Settings()
    
    # Check file type
    mime = magic.Magic(mime=True)
    file_type = mime.from_file(file_path)
    if file_type not in settings.ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {settings.ALLOWED_VIDEO_TYPES}"
        )
    
    # Check file size
    file_size = Path(file_path).stat().st_size
    if file_size > settings.MAX_VIDEO_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.MAX_VIDEO_SIZE/1024/1024}MB"
        )
    
    # Check duration
    with VideoFileClip(file_path) as clip:
        duration = clip.duration
        if duration < settings.MIN_VIDEO_DURATION:
            raise HTTPException(
                status_code=400,
                detail=f"Video too short. Minimum duration: {settings.MIN_VIDEO_DURATION}s"
            )
        if duration > settings.MAX_VIDEO_DURATION:
            raise HTTPException(
                status_code=400,
                detail=f"Video too long. Maximum duration: {settings.MAX_VIDEO_DURATION}s"
            )