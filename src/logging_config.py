"""
Logging configuration for the Dynamic Asset Pricing Model.
Provides structured logging with different levels and output formats.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


class ModelLoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter for adding context to log messages."""
    
    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        super().__init__(logger, extra or {})
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log message and add extra context."""
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        # Add adapter's extra fields
        kwargs['extra'].update(self.extra)
        
        return msg, kwargs
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics."""
        extra = {
            'operation': operation,
            'duration_seconds': duration,
            'performance_metric': True
        }
        extra.update(kwargs)
        self.info(f"Performance: {operation} completed in {duration:.4f}s", extra=extra)
    
    def log_data_quality(self, dataset: str, quality_metrics: Dict[str, Any]):
        """Log data quality metrics."""
        extra = {
            'dataset': dataset,
            'data_quality': True,
            **quality_metrics
        }
        self.info(f"Data quality check for {dataset}", extra=extra)
    
    def log_model_metrics(self, model_name: str, metrics: Dict[str, Any]):
        """Log model performance metrics."""
        extra = {
            'model_name': model_name,
            'model_metrics': True,
            **metrics
        }
        self.info(f"Model metrics for {model_name}", extra=extra)


class LoggingManager:
    """Manages logging configuration and setup."""
    
    def __init__(self, log_dir: Optional[str] = None, log_level: str = "INFO"):
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_level = getattr(logging, log_level.upper())
        self.loggers: Dict[str, logging.Logger] = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging configuration."""
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler for general logs
        general_log_file = self.log_dir / "model.log"
        file_handler = logging.handlers.RotatingFileHandler(
            general_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(console_formatter)
        root_logger.addHandler(file_handler)
        
        # JSON file handler for structured logs
        json_log_file = self.log_dir / "model_structured.log"
        json_handler = logging.handlers.RotatingFileHandler(
            json_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        json_handler.setLevel(self.log_level)
        json_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(json_handler)
        
        # Error-only handler
        error_log_file = self.log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(console_formatter)
        root_logger.addHandler(error_handler)
        
        # Performance log handler
        perf_log_file = self.log_dir / "performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(JSONFormatter())
        
        # Create performance logger
        perf_logger = logging.getLogger('performance')
        perf_logger.addHandler(perf_handler)
        perf_logger.propagate = False
    
    def get_logger(self, name: str, extra_context: Optional[Dict[str, Any]] = None) -> ModelLoggerAdapter:
        """Get a logger with optional extra context."""
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        
        return ModelLoggerAdapter(self.loggers[name], extra_context)
    
    def set_log_level(self, level: str):
        """Set logging level for all handlers."""
        self.log_level = getattr(logging, level.upper())
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        for handler in root_logger.handlers:
            if handler.level != logging.ERROR:  # Don't change error handler level
                handler.setLevel(self.log_level)


# Global logging manager instance
_logging_manager = None

def setup_logging(log_dir: Optional[str] = None, log_level: str = "INFO") -> LoggingManager:
    """Set up logging configuration."""
    global _logging_manager
    _logging_manager = LoggingManager(log_dir, log_level)
    return _logging_manager

def get_logger(name: str, extra_context: Optional[Dict[str, Any]] = None) -> ModelLoggerAdapter:
    """Get a logger instance."""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
    return _logging_manager.get_logger(name, extra_context)

def log_function_call(func):
    """Decorator to log function calls with timing."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.now()
        
        try:
            logger.debug(f"Starting {func.__name__}")
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.log_performance(func.__name__, duration)
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error in {func.__name__} after {duration:.4f}s: {str(e)}", exc_info=True)
            raise
    
    return wrapper


class LogContext:
    """Context manager for adding extra context to logs."""
    
    def __init__(self, logger: ModelLoggerAdapter, **context):
        self.logger = logger
        self.context = context
        self.original_extra = logger.extra.copy()
    
    def __enter__(self):
        self.logger.extra.update(self.context)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.extra = self.original_extra
        if exc_type is not None:
            self.logger.error(f"Exception in context: {exc_val}", exc_info=True)


# Convenience functions for common logging patterns
def log_data_load(logger: ModelLoggerAdapter, dataset: str, records: int, duration: float):
    """Log data loading operation."""
    logger.log_performance(
        f"load_{dataset}",
        duration,
        dataset=dataset,
        records_loaded=records,
        records_per_second=records/duration if duration > 0 else 0
    )

def log_model_fit(logger: ModelLoggerAdapter, model_name: str, duration: float, **metrics):
    """Log model fitting operation."""
    logger.log_performance(
        f"fit_{model_name}",
        duration,
        model_name=model_name,
        **metrics
    )

def log_prediction(logger: ModelLoggerAdapter, model_name: str, n_predictions: int, duration: float):
    """Log prediction operation."""
    logger.log_performance(
        f"predict_{model_name}",
        duration,
        model_name=model_name,
        predictions_made=n_predictions,
        predictions_per_second=n_predictions/duration if duration > 0 else 0
    )