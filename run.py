# run.py
import uvicorn
from beavervision.utils.logger import logger
from beavervision.utils.monitoring import init_monitoring

def main():
    """
    Main entry point for the application.
    """
    try:
        # Initialize monitoring
        init_monitoring()
        logger.info("Starting BeaverVision API server...")
        
        # Run the server
        uvicorn.run(
            "beavervision.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise

if __name__ == "__main__":
    main()