import logging
import os
from typing import Optional

def setup_logging(log_dir: Optional[str] = None, level: int = logging.INFO):
    """
    Set up logging configuration
    
    Args:
        log_dir: Directory to save log files (if None, only console logging is used)
        level: Logging level
    """
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add file handler if log_dir is provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, 'evaluation.log'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
