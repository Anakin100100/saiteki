import time
from shinka.core import EvolutionRunner
from loguru import logger
from worker.shutdown_service import ShutdownService
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

def main():
    logger.info("Starting saiteki worker")
    shutdown_service = ShutdownService()
    print("Hello from saiteki-worker!")
    
    logger.info("Worker is running. Press Ctrl+C to stop gracefully.")
    
    try:
        while not shutdown_service.should_shutdown():
            job_config = LocalJobConfig(eval_program_path="evaluate.py")
            db_config = DatabaseConfig()
            evo_config = EvolutionConfig(init_program_path="initial.py",)

            # Run evolution with defaults
            runner = EvolutionRunner(
                evo_config=evo_config,
                job_config=job_config,
                db_config=db_config,
            )
            runner.run()
            
        logger.info("Shutdown signal received, stopping worker...")
        
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        shutdown_service.shutdown()
    
    logger.info("Saiteki worker stopped")

if __name__ == "__main__":
    main()
