"""
Configuration Module for Reasoning Components

This module provides configuration settings and utilities for the reasoning components.
It centralizes configuration parameters, logging setup, and common utility functions.
"""

import os
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
import traceback

# Create logs directory if it doesn't exist
os.makedirs("logs/reasoning", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/reasoning/reasoning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("reasoning")

# Path for models and other resources
RESOURCES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")
os.makedirs(RESOURCES_DIR, exist_ok=True)

# Path for caching
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Default model settings
DEFAULT_MODEL_SETTINGS = {
    "model": "deepseek-ai/deepseek-v1",
    "temperature": 0.2,
    "max_tokens": 2000,
    "api_base": os.getenv("DEEPSEEK_API_BASE") or "https://api.deepseek.com/v1",
    "api_key": os.getenv("DEEPSEEK_API_KEY"),
    # Fallback to OpenAI if DeepSeek is not available
    "fallback_model": "gpt-4o-mini",
    "fallback_api_base": "https://api.openai.com/v1",
    "fallback_api_key": os.getenv("OPENAI_API_KEY"),
    "use_fallback": False  # Set to True to force using fallback
}

# Prompts file
PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")

def load_prompts() -> Dict[str, str]:
    """
    Load prompts from the prompts.json file.
    
    Returns:
        Dict[str, str]: Dictionary of prompt templates
    """
    try:
        if os.path.exists(PROMPTS_FILE):
            with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"Prompts file not found at {PROMPTS_FILE}, using default prompts")
            return {
                "analyze_query": "Please analyze this question: '{query}'\n\nProvide an analysis with the following information:\n1. Complexity level (simple, moderate, complex)\n2. Question type (factoid, comparative, exploratory, causal, etc.)\n3. Key concepts and entities mentioned\n4. Whether the question requires breaking down into sub-questions\n5. Domain or knowledge areas relevant to the question\n\nFormat your response as a structured analysis that can be easily parsed.",
                "decompose_question": "Please break down the following complex question into smaller, more manageable sub-questions that will help answer the original question when combined.\n\nOriginal question: {query}\n\nProvide 3-5 sub-questions that:\n1. Are simpler and more focused than the original\n2. Cover different aspects needed to fully answer the original\n3. Are arranged in a logical sequence\n4. Include dependencies between questions if applicable\n\nFor each sub-question, explain its relevance to the original question.",
                "integrate_results": "I've broken down a complex question into sub-questions and found information for each part. Please help me integrate this information into a comprehensive answer to the original question.\n\nOriginal question: {query}\n\nHere are the sub-questions and their answers:\n\n{subquestions_and_answers}\n\nPlease provide a well-structured, cohesive response to the original question, integrating all relevant information from the sub-questions."
            }
    except Exception as e:
        logger.error(f"Error loading prompts: {e}")
        logger.error(traceback.format_exc())
        return {}

# Load prompt templates
PROMPTS = load_prompts()

class ReasoningError(Exception):
    """Custom exception for reasoning-related errors."""
    
    def __init__(self, message: str, step: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        """
        Initialize the error.
        
        Args:
            message (str): Error message
            step (Optional[str]): The reasoning step where the error occurred
            data (Optional[Dict[str, Any]]): Additional data about the error
        """
        self.step = step
        self.data = data or {}
        super().__init__(message)
        
        # Log the error
        logger.error(f"ReasoningError in step '{step}': {message}")
        if data:
            logger.error(f"Error data: {json.dumps(data, indent=2)}") 