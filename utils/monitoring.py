from prometheus_client import Counter, Histogram, start_http_server
import time
from functools import wraps
from ..config.settings import Settings

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

def init_monitoring():
    if Settings().ENABLE_METRICS:
        start_http_server(Settings().PROMETHEUS_PORT)

def monitor_timing(process_type: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with PROCESSING_TIME.labels(process_type).time():
                return await func(*args, **kwargs)
        return wrapper
    return decorator