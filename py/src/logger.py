# py/src/logger.py
import logging
import sys
import io

# Create an in-memory stream to capture log messages
log_stream = io.StringIO()

def setup_logger():
    """
    Configures and returns a logger that writes to both stderr and an in-memory stream.
    This allows logs to be captured for inclusion in the final JSON output.
    """
    logger = logging.getLogger("fangzhen_be_py")
    logger.setLevel(logging.DEBUG)
    
    # Avoid adding handlers if they already exist
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Handler for stderr (for live debugging if running manually)
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
        
        # Handler for the in-memory stream
        string_handler = logging.StreamHandler(log_stream)
        string_handler.setFormatter(formatter)
        logger.addHandler(string_handler)
        
    return logger

def get_log_contents() -> str:
    """
    Retrieves the full contents of the in-memory log stream.
    """
    return log_stream.getvalue()

# Create a single logger instance to be used by all modules
log = setup_logger()
