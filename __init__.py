# beavervision/__init__.py
"""
BeaverVision - Real-time lip synchronization API
"""
from pathlib import Path

# Package metadata
__version__ = '0.1.0'
__author__ = 'Lord Amdal'

# Ensure required directories exist
PACKAGE_ROOT = Path(__file__).parent
LOGS_DIR = PACKAGE_ROOT / 'logs'
MODELS_DIR = PACKAGE_ROOT / 'models'
TEMP_DIR = PACKAGE_ROOT / 'temp'

for directory in [LOGS_DIR, MODELS_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)