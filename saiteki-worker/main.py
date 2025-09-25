import time
import json
import asyncio
from nats.aio.client import Client as NATS
from loguru import logger
from worker.shutdown_service import ShutdownService
from worker.models import OptimizationTask
from worker.task_service import TaskService

async def main():
    logger.info("Starting saiteki worker")
    shutdown_service = ShutdownService()
    task_service = TaskService()
    print("Hello from saiteki-worker!")
    
    # Connect to NATS
    nc = NATS()
    try:
        await nc.connect("nats://127.0.0.1:4222")
        logger.info("Connected to NATS server")
    except Exception as e:
        logger.error(f"Failed to connect to NATS: {e}")
        return
    
    # Subscribe to the subject
    sub = await nc.subscribe("created_optimization_tasks")
    logger.info("Subscribed to created_optimization_tasks")
    
    logger.info("Worker is running. Press Ctrl+C to stop gracefully.")
    
    try:
        while not shutdown_service.should_shutdown():
            try:
                # Try to get a message with 1 second timeout
                msg = await sub.next_msg(timeout=1.0)
                
                # Parse and log the message
                try:
                    message_data = json.loads(msg.data.decode())
                    logger.info(f"Received message: {message_data}")
                    
                    # Parse into Pydantic model
                    task = OptimizationTask(**message_data)
                    logger.info(f"Processing optimization task {task.id}")
                    
                    # Run the optimization task (this is synchronous but contains async API calls)
                    task_service.run_optimization_task(task)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message JSON: {e}")
                except Exception as e:
                    logger.error(f"Error processing task: {e}")
                    
            except Exception:
                # No message available or timeout, sleep for 1 second
                await asyncio.sleep(1)
            
        logger.info("Shutdown signal received, stopping worker...")
        
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        shutdown_service.shutdown()
    finally:
        await nc.close()
    
    logger.info("Saiteki worker stopped")

if __name__ == "__main__":
    asyncio.run(main())
