# beavervision/utils/monitoring.py
import time
from functools import wraps
from prometheus_client import Counter, Histogram, start_http_server
from beavervision.config import settings
from beavervision.utils.logger import setup_logger

logger = setup_logger(__name__)

# Metrics
PROCESSING_TIME = Histogram(
    'video_processing_seconds',
    'Time spent processing video',
    ['process_type']
)

REQUESTS_TOTAL = Counter(
    'requests_total',
    'Total number of requests',
    ['endpoint', 'status']
)

ERROR_COUNTER = Counter(
    'errors_total',
    'Total number of errors',
    ['error_type']
)

# GPU metrics (if available)
try:
    import torch
    GPU_MEMORY_USAGE = Gauge(
        'gpu_memory_usage_bytes',
        'GPU memory usage in bytes',
        ['device']
    )
    
    def update_gpu_metrics():
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i)
                GPU_MEMORY_USAGE.labels(device=f'cuda:{i}').set(memory_allocated)
except ImportError:
    logger.warning("torch not available, GPU metrics disabled")

def init_monitoring():
    """Initialize monitoring server if enabled in settings."""
    try:
        if settings.ENABLE_METRICS:
            start_http_server(settings.PROMETHEUS_PORT)
            logger.info(f"Monitoring server started on port {settings.PROMETHEUS_PORT}")
    except Exception as e:
        logger.error(f"Failed to start monitoring server: {str(e)}")

def monitor_timing(process_type: str):
    """
    Decorator to monitor the execution time of async functions.
    
    Args:
        process_type (str): Type of process being monitored
        
    Returns:
        wrapper: Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                with PROCESSING_TIME.labels(process_type).time():
                    return await func(*args, **kwargs)
            except Exception as e:
                ERROR_COUNTER.labels(error_type=process_type).inc()
                raise
        return wrapper
    return decorator