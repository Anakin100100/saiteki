import time
from shinka.core import EvolutionRunner
from loguru import logger
from worker.shutdown_service import ShutdownService

def main():
    logger.info("Starting saiteki worker")
    shutdown_service = ShutdownService()
    print("Hello from saiteki-worker!")
    
    logger.info("Worker is running. Press Ctrl+C to stop gracefully.")
    
    try:
        while not shutdown_service.should_shutdown():
            # Main worker logic here
            # For now, just sleep to simulate work
            time.sleep(1)
            
        logger.info("Shutdown signal received, stopping worker...")
        
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        shutdown_service.shutdown()
    
    logger.info("Saiteki worker stopped")

if __name__ == "__main__":
    main()
