import signal
import threading
from loguru import logger


class ShutdownService:
    def __init__(self):
        self._shutdown_event = threading.Event()
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_event.set()
    
    def should_shutdown(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_event.is_set()
    
    def shutdown(self):
        """Manually trigger shutdown."""
        logger.info("Manual shutdown requested")
        self._shutdown_event.set()
    
    def wait_for_shutdown(self, timeout=None):
        """Wait for shutdown signal."""
        return self._shutdown_event.wait(timeout)
