# beavervision/utils/validators.py
from pathlib import Path
import magic
from fastapi import HTTPException
from moviepy.editor import VideoFileClip
from beavervision.config import settings
from beavervision.utils.logger import setup_logger

logger = setup_logger(__name__)

def validate_video(file_path: str):
    """
    Validate video file format, size, and duration.
    
    Args:
        file_path (str): Path to the video file
        
    Raises:
        HTTPException: If validation fails
    """
    try:
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            raise HTTPException(
                status_code=400,
                detail="Video file not found"
            )
        
        # Check file type
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(str(file_path))
        if file_type not in settings.ALLOWED_VIDEO_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file_type}. Allowed types: {settings.ALLOWED_VIDEO_TYPES}"
            )
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > settings.MAX_VIDEO_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large: {file_size/1024/1024:.1f}MB. Maximum size: {settings.MAX_VIDEO_SIZE/1024/1024}MB"
            )
        
        # Check duration
        with VideoFileClip(str(file_path)) as clip:
            duration = clip.duration
            if duration < settings.MIN_VIDEO_DURATION:
                raise HTTPException(
                    status_code=400,
                    detail=f"Video too short: {duration:.1f}s. Minimum duration: {settings.MIN_VIDEO_DURATION}s"
                )
            if duration > settings.MAX_VIDEO_DURATION:
                raise HTTPException(
                    status_code=400,
                    detail=f"Video too long: {duration:.1f}s. Maximum duration: {settings.MAX_VIDEO_DURATION}s"
                )
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video validation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error validating video: {str(e)}"
        )