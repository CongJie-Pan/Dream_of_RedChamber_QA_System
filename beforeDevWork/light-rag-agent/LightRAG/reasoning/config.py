"""
Configuration module for LightRAG reasoning agent.

This module contains configuration settings, constants, and utilities
for the reasoning agent implementation, including logging setup and
error handling mechanisms.
"""

import os
import json
import logging
import logging.config
from typing import Dict, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Constants
DEFAULT_API_BASE = "https://api.deepseek.com"
DEFAULT_MODEL_NAME = "deepseek-r1-chat"
API_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # exponential backoff multiplier

# Reasoning Parameters
MAX_SUB_QUESTIONS = 5
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_K = 5
DEFAULT_MAX_TOKENS = 1024

# Paths
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": os.path.join(LOG_DIR, "reasoning_agent.log"),
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 5,
            "encoding": "utf8"
        },
        "error_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": os.path.join(LOG_DIR, "reasoning_errors.log"),
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 5,
            "encoding": "utf8"
        }
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["console", "file_handler", "error_file_handler"],
            "level": "INFO",
            "propagate": True
        },
        "reasoning": {
            "handlers": ["console", "file_handler", "error_file_handler"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}

# Initialize logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("reasoning")

# Error Classes
class DeepSeekAPIError(Exception):
    """Exception raised for errors in the DeepSeek API interaction."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict[str, Any]] = None):
        """
        Initialize the DeepSeek API error.
        
        Args:
            message (str): Error message.
            status_code (Optional[int]): HTTP status code if available.
            response (Optional[Dict[str, Any]]): Raw API response if available.
        """
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)
        
    def __str__(self) -> str:
        """Return string representation of the error with status code if available."""
        if self.status_code:
            return f"{self.message} (Status Code: {self.status_code})"
        return self.message

class ReasoningError(Exception):
    """Exception raised for errors in the reasoning process."""
    
    def __init__(self, message: str, step: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        """
        Initialize the reasoning error.
        
        Args:
            message (str): Error message.
            step (Optional[str]): The reasoning step where the error occurred.
            data (Optional[Dict[str, Any]]): Additional data related to the error.
        """
        self.message = message
        self.step = step
        self.data = data
        msg = f"{message}"
        if step:
            msg += f" (Step: {step})"
        super().__init__(msg)

# Configuration Utility Functions
def get_api_key() -> str:
    """
    Get the DeepSeek API key from environment variables.
    
    Returns:
        str: The API key.
        
    Raises:
        ValueError: If the API key is not found.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("DeepSeek API key not found in environment variables")
        raise ValueError("DeepSeek API key not found. Please set the DEEPSEEK_API_KEY environment variable.")
    return api_key

def get_api_base() -> str:
    """
    Get the DeepSeek API base URL from environment variables or use default.
    
    Returns:
        str: The API base URL.
    """
    return os.getenv("DEEPSEEK_API_BASE", DEFAULT_API_BASE)

def get_model_name() -> str:
    """
    Get the DeepSeek model name from environment variables or use default.
    
    Returns:
        str: The model name.
    """
    return os.getenv("DEEPSEEK_MODEL_NAME", DEFAULT_MODEL_NAME)

def load_custom_prompts() -> Dict[str, str]:
    """
    Load custom prompts from a JSON file if available.
    
    Returns:
        Dict[str, str]: Dictionary of custom prompts.
    """
    prompt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")
    
    if os.path.exists(prompt_file):
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load custom prompts: {e}")
    
    # Return empty dict if file doesn't exist or loading fails
    return {} 