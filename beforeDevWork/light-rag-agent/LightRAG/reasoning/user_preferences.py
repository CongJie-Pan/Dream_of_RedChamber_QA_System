"""
User Preferences Management Module

This module provides functionality for storing, retrieving, and managing
user preferences across sessions for the LightRAG reasoning system.

Features:
- Store user interface preferences
- Save customized reasoning parameters
- Manage visualization settings
- Support for multiple named user profiles
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
from .config import logger


@dataclass
class UserPreferences:
    """
    Container for user preferences settings.
    
    This dataclass stores all customizable user preferences including
    visualization options, reasoning parameters, and UI settings.
    
    Attributes:
        user_id (str): User identifier
        reasoning_preferences (Dict[str, Any]): Customized reasoning parameters
        visualization_preferences (Dict[str, Any]): Visualization settings
        ui_preferences (Dict[str, Any]): User interface settings
        last_updated (str): Timestamp of last update
    """
    user_id: str
    reasoning_preferences: Dict[str, Any] = field(default_factory=lambda: {
        "max_sub_questions": 5,
        "temperature": 0.2,
        "enable_parallel_processing": True,
        "retrieval_method": "hybrid",
        "similarity_threshold": 0.6
    })
    visualization_preferences: Dict[str, Any] = field(default_factory=lambda: {
        "default_format": "html",
        "show_dependency_graph": True,
        "show_timeline": True,
        "expanded_by_default": False,
        "color_scheme": "default"
    })
    ui_preferences: Dict[str, Any] = field(default_factory=lambda: {
        "language": "zh-TW",
        "show_debug_info": False,
        "auto_expand_reasoning": False,
        "show_confidence_scores": True,
        "theme": "light"
    })
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


class UserPreferencesManager:
    """
    Manager for user preferences storage and retrieval.
    
    This class provides methods for storing, retrieving, and managing
    user preferences across sessions, supporting multiple named profiles.
    
    Attributes:
        storage_dir (str): Directory to store user preferences files
        default_preferences (UserPreferences): Default user preferences
        current_user_id (str): Currently active user ID
        preferences_cache (Dict[str, UserPreferences]): Cache of loaded preferences
    """
    
    def __init__(self, storage_dir: Optional[str] = None, default_user_id: str = "default_user"):
        """
        Initialize the preferences manager.
        
        Args:
            storage_dir (Optional[str]): Directory to store preferences. If None, uses default.
            default_user_id (str): User ID to use by default
        """
        # Determine storage directory
        if storage_dir:
            self.storage_dir = storage_dir
        else:
            # Default path in user's home directory
            self.storage_dir = os.path.join(os.path.expanduser("~"), ".lightrag", "preferences")
        
        # Ensure directory exists
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Set up default preferences
        self.default_preferences = UserPreferences(user_id=default_user_id)
        
        # Initialize with the default user
        self.current_user_id = default_user_id
        
        # Cache of loaded preferences
        self.preferences_cache = {}
        
        # Load default user preferences if they exist
        self.load_preferences(default_user_id)
        
        logger.info(f"UserPreferencesManager initialized with storage directory: {self.storage_dir}")
    
    def _get_preferences_path(self, user_id: str) -> str:
        """
        Get the file path for a user's preferences.
        
        Args:
            user_id (str): User identifier
            
        Returns:
            str: Path to the user's preferences file
        """
        # Sanitize user_id to be safe for filenames
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in "_-")
        if not safe_user_id:
            safe_user_id = "unknown_user"
            
        return os.path.join(self.storage_dir, f"{safe_user_id}_preferences.json")
    
    def load_preferences(self, user_id: Optional[str] = None) -> UserPreferences:
        """
        Load preferences for a specific user.
        
        Args:
            user_id (Optional[str]): User identifier. If None, uses current user.
            
        Returns:
            UserPreferences: The loaded user preferences
        """
        # Use specified user_id or current user
        user_id = user_id or self.current_user_id
        
        # Return cached preferences if available
        if user_id in self.preferences_cache:
            return self.preferences_cache[user_id]
        
        # Get file path
        file_path = self._get_preferences_path(user_id)
        
        # If file exists, load preferences
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Create UserPreferences from loaded data
                preferences = UserPreferences(
                    user_id=user_id,
                    reasoning_preferences=data.get('reasoning_preferences', {}),
                    visualization_preferences=data.get('visualization_preferences', {}),
                    ui_preferences=data.get('ui_preferences', {}),
                    last_updated=data.get('last_updated', datetime.now().isoformat())
                )
                
                # Merge with defaults to ensure all fields are present
                preferences = self._merge_with_defaults(preferences)
                
                # Cache and return preferences
                self.preferences_cache[user_id] = preferences
                logger.info(f"Loaded preferences for user: {user_id}")
                return preferences
                
            except Exception as e:
                # Log error and return default preferences
                logger.error(f"Error loading preferences for user {user_id}: {e}")
                logger.info(f"Using default preferences for user: {user_id}")
                
                # Cache default preferences for this user
                default_for_user = UserPreferences(user_id=user_id)
                self.preferences_cache[user_id] = default_for_user
                return default_for_user
        else:
            # No existing preferences file, use defaults
            logger.info(f"No existing preferences found for user: {user_id}. Using defaults.")
            
            # Cache default preferences for this user
            default_for_user = UserPreferences(user_id=user_id)
            self.preferences_cache[user_id] = default_for_user
            return default_for_user
    
    def _merge_with_defaults(self, preferences: UserPreferences) -> UserPreferences:
        """
        Merge loaded preferences with defaults to ensure all fields are present.
        
        Args:
            preferences (UserPreferences): The preferences to merge
            
        Returns:
            UserPreferences: Merged preferences
        """
        # Get default values
        default_reasoning = self.default_preferences.reasoning_preferences
        default_visualization = self.default_preferences.visualization_preferences
        default_ui = self.default_preferences.ui_preferences
        
        # Update with loaded values (keeping defaults for missing fields)
        merged_reasoning = {**default_reasoning, **preferences.reasoning_preferences}
        merged_visualization = {**default_visualization, **preferences.visualization_preferences}  
        merged_ui = {**default_ui, **preferences.ui_preferences}
        
        # Return new preferences object with merged values
        return UserPreferences(
            user_id=preferences.user_id,
            reasoning_preferences=merged_reasoning,
            visualization_preferences=merged_visualization,
            ui_preferences=merged_ui,
            last_updated=preferences.last_updated
        )
    
    def save_preferences(self, preferences: Optional[UserPreferences] = None,
                       user_id: Optional[str] = None) -> bool:
        """
        Save user preferences to storage.
        
        Args:
            preferences (Optional[UserPreferences]): Preferences to save. If None, saves current.
            user_id (Optional[str]): User identifier. If None, uses preferences user_id.
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        # Determine preferences and user_id to use
        if preferences is None:
            # If no preferences provided, get current user's preferences
            user_id = user_id or self.current_user_id
            preferences = self.load_preferences(user_id)
        else:
            # If preferences provided, use its user_id unless overridden
            user_id = user_id or preferences.user_id
        
        # Update the timestamp
        preferences.last_updated = datetime.now().isoformat()
        
        # Get file path
        file_path = self._get_preferences_path(user_id)
        
        try:
            # Convert preferences to dictionary and save
            preferences_dict = asdict(preferences)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(preferences_dict, f, indent=2)
            
            # Update cache
            self.preferences_cache[user_id] = preferences
            
            logger.info(f"Saved preferences for user: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving preferences for user {user_id}: {e}")
            return False
    
    def update_preferences(self, updates: Dict[str, Any], user_id: Optional[str] = None) -> bool:
        """
        Update specific preference settings.
        
        Args:
            updates (Dict[str, Any]): Dictionary of preference updates
            user_id (Optional[str]): User identifier. If None, uses current user.
            
        Returns:
            bool: True if updated successfully, False otherwise
        """
        # Get user_id to use
        user_id = user_id or self.current_user_id
        
        # Load current preferences
        preferences = self.load_preferences(user_id)
        
        # Apply updates to the appropriate sections
        for section, section_updates in updates.items():
            if section == "reasoning_preferences":
                preferences.reasoning_preferences.update(section_updates)
            elif section == "visualization_preferences":
                preferences.visualization_preferences.update(section_updates)
            elif section == "ui_preferences":
                preferences.ui_preferences.update(section_updates)
            else:
                logger.warning(f"Unknown preferences section: {section}")
        
        # Save updated preferences
        return self.save_preferences(preferences, user_id)
    
    def get_preference(self, section: str, key: str, user_id: Optional[str] = None) -> Any:
        """
        Get a specific preference value.
        
        Args:
            section (str): Preference section ('reasoning', 'visualization', 'ui')
            key (str): Preference key
            user_id (Optional[str]): User identifier. If None, uses current user.
            
        Returns:
            Any: The preference value or None if not found
        """
        # Get user_id to use
        user_id = user_id or self.current_user_id
        
        # Load preferences
        preferences = self.load_preferences(user_id)
        
        # Get the preference section
        if section == "reasoning":
            section_dict = preferences.reasoning_preferences
        elif section == "visualization":
            section_dict = preferences.visualization_preferences
        elif section == "ui":
            section_dict = preferences.ui_preferences
        else:
            logger.warning(f"Unknown preferences section: {section}")
            return None
        
        # Return the preference value or None if not found
        return section_dict.get(key)
    
    def set_current_user(self, user_id: str) -> bool:
        """
        Set the current active user.
        
        Args:
            user_id (str): User identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Load preferences for the user to ensure they exist
        try:
            self.load_preferences(user_id)
            self.current_user_id = user_id
            logger.info(f"Current user set to: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error setting current user to {user_id}: {e}")
            return False
    
    def get_current_user_id(self) -> str:
        """
        Get the current user ID.
        
        Returns:
            str: Current user ID
        """
        return self.current_user_id
    
    def get_all_user_ids(self) -> List[str]:
        """
        Get a list of all users with saved preferences.
        
        Returns:
            List[str]: List of user IDs
        """
        user_ids = []
        
        # List all files in the storage directory
        for filename in os.listdir(self.storage_dir):
            if filename.endswith("_preferences.json"):
                # Extract user_id from filename
                user_id = filename.replace("_preferences.json", "")
                user_ids.append(user_id)
        
        return user_ids
    
    def create_user_profile(self, user_id: str) -> bool:
        """
        Create a new user profile with default preferences.
        
        Args:
            user_id (str): User identifier
            
        Returns:
            bool: True if created successfully, False otherwise
        """
        # Create default preferences for the user
        preferences = UserPreferences(user_id=user_id)
        
        # Save preferences
        success = self.save_preferences(preferences)
        
        if success:
            logger.info(f"Created new user profile: {user_id}")
        
        return success
    
    def delete_user_profile(self, user_id: str) -> bool:
        """
        Delete a user profile and associated preferences.
        
        Args:
            user_id (str): User identifier to delete
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        # Check if trying to delete the default user
        if user_id == "default_user":
            logger.error("Cannot delete the default user profile")
            return False
        
        # Get file path
        file_path = self._get_preferences_path(user_id)
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"User profile not found: {user_id}")
            return False
        
        try:
            # Remove the file
            os.remove(file_path)
            
            # Remove from cache if present
            if user_id in self.preferences_cache:
                del self.preferences_cache[user_id]
            
            # If this was the current user, switch to default
            if user_id == self.current_user_id:
                self.current_user_id = "default_user"
            
            logger.info(f"Deleted user profile: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting user profile {user_id}: {e}")
            return False
    
    def import_preferences(self, file_path: str, user_id: Optional[str] = None) -> bool:
        """
        Import preferences from a JSON file.
        
        Args:
            file_path (str): Path to the preferences JSON file
            user_id (Optional[str]): User identifier to assign preferences to
            
        Returns:
            bool: True if imported successfully, False otherwise
        """
        try:
            # Read the JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Determine user_id to use
            target_user_id = user_id or data.get('user_id', self.current_user_id)
            
            # Create UserPreferences from loaded data
            preferences = UserPreferences(
                user_id=target_user_id,
                reasoning_preferences=data.get('reasoning_preferences', {}),
                visualization_preferences=data.get('visualization_preferences', {}),
                ui_preferences=data.get('ui_preferences', {}),
                last_updated=datetime.now().isoformat()  # Use current time for import
            )
            
            # Save preferences
            success = self.save_preferences(preferences)
            
            if success:
                logger.info(f"Imported preferences for user: {target_user_id}")
            
            return success
        except Exception as e:
            logger.error(f"Error importing preferences: {e}")
            return False
    
    def export_preferences(self, user_id: Optional[str] = None, file_path: Optional[str] = None) -> Optional[str]:
        """
        Export preferences to a JSON file.
        
        Args:
            user_id (Optional[str]): User identifier. If None, uses current user.
            file_path (Optional[str]): Path to save the file. If None, generates path.
            
        Returns:
            Optional[str]: Path to the saved file or None if export failed
        """
        # Get user_id to use
        user_id = user_id or self.current_user_id
        
        # Load preferences
        preferences = self.load_preferences(user_id)
        
        # Determine file path
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{user_id}_preferences_{timestamp}.json"
            file_path = os.path.join(self.storage_dir, "exports", filename)
            
            # Ensure exports directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        try:
            # Convert preferences to dictionary and save
            preferences_dict = asdict(preferences)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(preferences_dict, f, indent=2)
            
            logger.info(f"Exported preferences for user {user_id} to: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error exporting preferences for user {user_id}: {e}")
            return None
    
    def get_preferences_history(self, user_id: Optional[str] = None, 
                              max_entries: int = 10) -> List[Dict[str, Any]]:
        """
        Get history of preference changes for a user.
        
        This method requires an additional backup storage mechanism to be implemented.
        
        Args:
            user_id (Optional[str]): User identifier. If None, uses current user.
            max_entries (int): Maximum number of history entries to return
            
        Returns:
            List[Dict[str, Any]]: List of historical preference entries
        """
        # Get user_id to use
        user_id = user_id or self.current_user_id
        
        # Get history file path
        history_dir = os.path.join(self.storage_dir, "history")
        os.makedirs(history_dir, exist_ok=True)
        history_file = os.path.join(history_dir, f"{user_id}_history.json")
        
        if not os.path.exists(history_file):
            logger.info(f"No preference history found for user: {user_id}")
            return []
        
        try:
            # Read history file
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            # Return up to max_entries, most recent first
            return history[-max_entries:]
        except Exception as e:
            logger.error(f"Error reading preference history for user {user_id}: {e}")
            return []
    
    def reset_to_defaults(self, user_id: Optional[str] = None) -> bool:
        """
        Reset user preferences to default values.
        
        Args:
            user_id (Optional[str]): User identifier. If None, uses current user.
            
        Returns:
            bool: True if reset successfully, False otherwise
        """
        # Get user_id to use
        user_id = user_id or self.current_user_id
        
        try:
            # Create a new UserPreferences with default values
            default_preferences = UserPreferences(user_id=user_id)
            
            # Save the default preferences
            success = self.save_preferences(default_preferences)
            
            if success:
                logger.info(f"Reset preferences to defaults for user: {user_id}")
            
            return success
        except Exception as e:
            logger.error(f"Error resetting preferences for user {user_id}: {e}")
            return False 