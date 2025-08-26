"""
Centralized logging configuration for CLAIRE-RAG [BACKEND]
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Any, Dict
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;21m"
    blue = "\x1b[34;21m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.INFO: green + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_obj.update(record.extra_fields)
            
        return json.dumps(log_obj)


def setup_logger(
    name: str = "claire",
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    use_json: bool = False
) -> logging.Logger:
    """
    Set up a logger with file and console handlers
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        use_json: Use JSON format for file logging
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Console handler with colored output
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_to_file:
        # Main log file
        file_handler = RotatingFileHandler(
            LOGS_DIR / f"{name}.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        
        if use_json:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            )
        logger.addHandler(file_handler)
        
        # Error log file
        error_handler = RotatingFileHandler(
            LOGS_DIR / f"{name}_errors.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )
        logger.addHandler(error_handler)
        
        # Daily performance log
        perf_handler = TimedRotatingFileHandler(
            LOGS_DIR / f"{name}_performance.log",
            when='midnight',
            interval=1,
            backupCount=30
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(JSONFormatter())
        # Add filter to only log performance metrics
        perf_handler.addFilter(lambda record: 'performance' in record.getMessage().lower() or 'took' in record.getMessage().lower())
        logger.addHandler(perf_handler)
    
    return logger


# Create default loggers for different components
def get_logger(component: str = "general") -> logging.Logger:
    """Get a logger for a specific component"""
    loggers = {
        "general": setup_logger("claire", "INFO"),
        "models": setup_logger("claire.models", "INFO"),
        "api": setup_logger("claire.api", "INFO"),
        "rag": setup_logger("claire.rag", "INFO"),
        "performance": setup_logger("claire.performance", "INFO", use_json=True)
    }
    return loggers.get(component, loggers["general"])


# Utility function to log performance metrics
def log_performance(logger: logging.Logger, operation: str, duration: float, **kwargs):
    """Log performance metrics in a structured way"""
    extra_fields = {
        "operation": operation,
        "duration_seconds": duration,
        **kwargs
    }
    logger.info(
        f"Performance: {operation} took {duration:.2f}s",
        extra={"extra_fields": extra_fields}
    )


# Context manager for timing operations
class TimedOperation:
    """Context manager for timing and logging operations"""
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting: {self.operation}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"Completed: {self.operation} in {duration:.2f}s")
            log_performance(self.logger, self.operation, duration, status="success")
        else:
            self.logger.error(f"Failed: {self.operation} after {duration:.2f}s - {exc_val}")
            log_performance(self.logger, self.operation, duration, status="error", error=str(exc_val))