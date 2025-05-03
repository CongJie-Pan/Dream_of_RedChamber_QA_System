"""
Settings module for the LightRAG reasoning system.

This module provides utilities for managing user-configurable
settings, preferences, and controls for adjusting reasoning parameters.
"""

import os
import json
import copy
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from .config import logger

@dataclass
class ReasoningSettings:
    """
    User-configurable settings for the reasoning system.
    
    This class provides a centralized way to manage all configurable aspects
    of the reasoning process, including model parameters, decomposition
    settings, and visualization preferences.
    
    Attributes:
        temperature (float): Temperature parameter for LLM calls.
        max_sub_questions (int): Maximum number of sub-questions to generate.
        enable_caching (bool): Whether to enable result caching.
        enable_parallelism (bool): Whether to enable parallel processing.
        detailed_logging (bool): Whether to enable detailed logging.
        retrieval_parameters (Dict[str, Any]): Default retrieval parameters.
        visualization_format (str): Default visualization format.
        model_parameters (Dict[str, Any]): Additional model parameters.
    """
    
    temperature: float = 0.2
    max_sub_questions: int = 5
    enable_caching: bool = True
    enable_parallelism: bool = True
    detailed_logging: bool = True
    retrieval_parameters: Dict[str, Any] = field(default_factory=lambda: {
        "top_k": 5,
        "similarity_threshold": 0.6,
        "retrieval_method": "hybrid",
        "use_knowledge_graph": True
    })
    visualization_format: str = "html"
    model_parameters: Dict[str, Any] = field(default_factory=lambda: {
        "max_tokens": 1024,
        "streaming": False
    })

class SettingsManager:
    """
    Manager for user preferences and system settings.
    
    This class provides methods for loading, saving, and managing
    user-configurable settings for the reasoning system.
    
    Attributes:
        settings (ReasoningSettings): Current reasoning settings.
        settings_path (str): Path to the settings file.
        settings_history (List[Dict[str, Any]]): History of setting changes.
    """
    
    def __init__(self, settings_path: Optional[str] = None):
        """
        Initialize the settings manager.
        
        Args:
            settings_path (Optional[str]): Path to the settings file. If None, uses default.
        """
        # Set default settings
        self.settings = ReasoningSettings()
        
        # Determine settings file path
        if settings_path:
            self.settings_path = settings_path
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.settings_path = os.path.join(os.path.dirname(current_dir), "config", "user_settings.json")
            
            # Ensure config directory exists
            os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
        
        # Initialize settings history
        self.settings_history = []
        
        # Try to load settings from file
        self.load_settings()
    
    def load_settings(self) -> bool:
        """
        Load settings from the settings file.
        
        Returns:
            bool: True if settings were successfully loaded, False otherwise.
        """
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, "r", encoding="utf-8") as f:
                    settings_dict = json.load(f)
                    
                # Update settings if some fields have changed in the dataclass
                settings_copy = asdict(self.settings)
                settings_copy.update(settings_dict)
                
                # Create a new settings object with the updated values
                for key, value in settings_copy.items():
                    if hasattr(self.settings, key):
                        setattr(self.settings, key, value)
                
                logger.info(f"Settings loaded from {self.settings_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading settings: {e}")
                return False
        else:
            # No settings file found, create default settings
            self.save_settings()
            return False
    
    def save_settings(self) -> bool:
        """
        Save current settings to the settings file.
        
        Returns:
            bool: True if settings were successfully saved, False otherwise.
        """
        try:
            # Convert settings to dictionary and save
            settings_dict = asdict(self.settings)
            
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(settings_dict, f, indent=2)
            
            logger.info(f"Settings saved to {self.settings_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return False
    
    def update_settings(self, settings_update: Dict[str, Any]) -> bool:
        """
        Update settings with new values.
        
        Args:
            settings_update (Dict[str, Any]): Dictionary of settings to update.
            
        Returns:
            bool: True if settings were successfully updated, False otherwise.
        """
        # Record current settings in history
        self.settings_history.append(asdict(self.settings))
        
        # Apply updates to settings object
        for key, value in settings_update.items():
            if hasattr(self.settings, key):
                if isinstance(value, dict) and isinstance(getattr(self.settings, key), dict):
                    # For dictionary fields, update rather than replace
                    current_dict = getattr(self.settings, key)
                    current_dict.update(value)
                else:
                    setattr(self.settings, key, value)
            else:
                logger.warning(f"Unknown setting: {key}")
        
        # Save the updated settings
        return self.save_settings()
    
    def get_settings(self) -> Dict[str, Any]:
        """
        Get current settings as a dictionary.
        
        Returns:
            Dict[str, Any]: Current settings.
        """
        return asdict(self.settings)
    
    def get_setting(self, key: str) -> Any:
        """
        Get a specific setting value.
        
        Args:
            key (str): Setting key to retrieve.
            
        Returns:
            Any: The setting value or None if key doesn't exist.
        """
        if hasattr(self.settings, key):
            return getattr(self.settings, key)
        else:
            logger.warning(f"Unknown setting: {key}")
            return None
    
    def reset_to_defaults(self) -> bool:
        """
        Reset settings to default values.
        
        Returns:
            bool: True if settings were successfully reset, False otherwise.
        """
        # Record current settings in history
        self.settings_history.append(asdict(self.settings))
        
        # Create new settings with default values
        self.settings = ReasoningSettings()
        
        # Save the default settings
        return self.save_settings()
    
    def undo_last_change(self) -> bool:
        """
        Revert to previous settings.
        
        Returns:
            bool: True if settings were successfully reverted, False otherwise.
        """
        if not self.settings_history:
            logger.warning("No settings history available to undo.")
            return False
        
        # Get last settings from history
        previous_settings = self.settings_history.pop()
        
        # Apply previous settings
        for key, value in previous_settings.items():
            setattr(self.settings, key, value)
        
        # Save the reverted settings
        return self.save_settings()

class UserFeedbackManager:
    """
    Manager for collecting and processing user feedback on reasoning quality.
    
    This class provides methods for collecting, storing, and analyzing
    user feedback to improve reasoning quality over time.
    
    Attributes:
        feedback_path (str): Path to the feedback storage file.
        feedback_data (List[Dict[str, Any]]): List of collected feedback.
    """
    
    def __init__(self, feedback_path: Optional[str] = None):
        """
        Initialize the user feedback manager.
        
        Args:
            feedback_path (Optional[str]): Path to the feedback file. If None, uses default.
        """
        # Determine feedback file path
        if feedback_path:
            self.feedback_path = feedback_path
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.feedback_path = os.path.join(os.path.dirname(current_dir), "logs", "user_feedback.json")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.feedback_path), exist_ok=True)
        
        # Initialize feedback data
        self.feedback_data = []
        
        # Try to load existing feedback
        self.load_feedback()
    
    def load_feedback(self) -> bool:
        """
        Load feedback data from the feedback file.
        
        Returns:
            bool: True if feedback was successfully loaded, False otherwise.
        """
        if os.path.exists(self.feedback_path):
            try:
                with open(self.feedback_path, "r", encoding="utf-8") as f:
                    self.feedback_data = json.load(f)
                
                logger.info(f"Feedback data loaded from {self.feedback_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading feedback data: {e}")
                # Initialize empty feedback data
                self.feedback_data = []
                return False
        else:
            # No feedback file found, create an empty one
            self.feedback_data = []
            self.save_feedback()
            return False
    
    def save_feedback(self) -> bool:
        """
        Save feedback data to the feedback file.
        
        Returns:
            bool: True if feedback was successfully saved, False otherwise.
        """
        try:
            with open(self.feedback_path, "w", encoding="utf-8") as f:
                json.dump(self.feedback_data, f, indent=2)
            
            logger.info(f"Feedback data saved to {self.feedback_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving feedback data: {e}")
            return False
    
    def add_feedback(self, feedback: Dict[str, Any]) -> bool:
        """
        Add new user feedback.
        
        Args:
            feedback (Dict[str, Any]): Feedback data to add.
            
        Returns:
            bool: True if feedback was successfully added, False otherwise.
        """
        # Ensure we have the essential feedback fields
        required_fields = ["query", "rating", "feedback_text"]
        if not all(field in feedback for field in required_fields):
            logger.error(f"Feedback missing required fields: {required_fields}")
            return False
        
        # Add timestamp if not already present
        if "timestamp" not in feedback:
            from datetime import datetime
            feedback["timestamp"] = datetime.now().isoformat()
        
        # Add to feedback data
        self.feedback_data.append(feedback)
        
        # Save updated feedback
        return self.save_feedback()
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get statistics about collected feedback.
        
        Returns:
            Dict[str, Any]: Statistics about the feedback data.
        """
        if not self.feedback_data:
            return {"count": 0, "average_rating": 0.0, "ratings": {}}
        
        # Calculate average rating
        ratings = [fb.get("rating", 0) for fb in self.feedback_data]
        average_rating = sum(ratings) / len(ratings)
        
        # Count ratings by value
        rating_counts = {}
        for rating in ratings:
            rating_counts[rating] = rating_counts.get(rating, 0) + 1
        
        # Get common feedback themes
        feedback_texts = [fb.get("feedback_text", "") for fb in self.feedback_data]
        
        return {
            "count": len(self.feedback_data),
            "average_rating": average_rating,
            "ratings": rating_counts,
            "recent_feedback": self.feedback_data[-5:] if len(self.feedback_data) > 0 else []
        }
    
    def get_feedback_by_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Get feedback for a specific query.
        
        Args:
            query (str): Query to find feedback for.
            
        Returns:
            List[Dict[str, Any]]: List of feedback entries for the query.
        """
        # Match based on query text (exact match)
        exact_matches = [fb for fb in self.feedback_data if fb.get("query") == query]
        
        # Also try partial matching
        if not exact_matches:
            partial_matches = [fb for fb in self.feedback_data 
                             if query.lower() in fb.get("query", "").lower()]
            return partial_matches
        
        return exact_matches 