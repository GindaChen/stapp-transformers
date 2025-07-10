"""
Settings Management Module

This module handles all application settings with persistent storage.
"""

import streamlit as st
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

# Default settings
DEFAULT_SETTINGS = {
    'use_number_units': False,
    'cache_expiry_hours': 24,
    'max_recent_models': 10,
    'auto_refresh_on_settings_change': True,
    'sidebar_expanded': True
}

# Cache directory (will be set by app.py)
CACHE_DIR = None


def set_cache_dir(cache_dir: Path) -> None:
    """Set the cache directory for settings storage."""
    global CACHE_DIR
    CACHE_DIR = cache_dir


def set_cookie(key: str, value: Any, expires_days: int = 30) -> None:
    """Set a persistent value using local file storage."""
    if not CACHE_DIR:
        return
        
    try:
        # Create a user data directory
        user_data_dir = CACHE_DIR / "user_data"
        user_data_dir.mkdir(exist_ok=True)
        
        # Store data with expiry
        data = {
            'value': value,
            'expires': (datetime.now() + timedelta(days=expires_days)).isoformat()
        }
        
        # Save to file
        with open(user_data_dir / f"{key}.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Also store in session state if available
        try:
            if 'persistent_data' not in st.session_state:
                st.session_state.persistent_data = {}
            st.session_state.persistent_data[key] = data
        except:
            pass
        
    except Exception:
        # Fallback to session state only if available
        try:
            if 'persistent_data' not in st.session_state:
                st.session_state.persistent_data = {}
            st.session_state.persistent_data[key] = {
                'value': value,
                'expires': (datetime.now() + timedelta(days=expires_days)).isoformat()
            }
        except:
            pass


def get_cookie(key: str, default: Any = None) -> Any:
    """Get a persistent value from local file storage or session state."""
    if not CACHE_DIR:
        return default
        
    try:
        # First check session state if available
        try:
            if 'persistent_data' in st.session_state and key in st.session_state.persistent_data:
                data = st.session_state.persistent_data[key]
                expires = datetime.fromisoformat(data['expires'])
                
                if datetime.now() < expires:
                    return data['value']
        except:
            pass
        
        # Try to load from file
        user_data_dir = CACHE_DIR / "user_data"
        file_path = user_data_dir / f"{key}.json"
        
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            expires = datetime.fromisoformat(data['expires'])
            
            if datetime.now() < expires:
                # Store back in session state for faster access if available
                try:
                    if 'persistent_data' not in st.session_state:
                        st.session_state.persistent_data = {}
                    st.session_state.persistent_data[key] = data
                except:
                    pass
                
                return data['value']
            else:
                # File expired, remove it
                file_path.unlink()
        
    except Exception:
        pass
    
    return default


def clear_cookie(key: str) -> None:
    """Clear a persistent value."""
    if not CACHE_DIR:
        return
        
    try:
        # Remove from session state if available
        try:
            if 'persistent_data' in st.session_state and key in st.session_state.persistent_data:
                del st.session_state.persistent_data[key]
        except:
            pass
        
        # Remove file
        user_data_dir = CACHE_DIR / "user_data"
        file_path = user_data_dir / f"{key}.json"
        if file_path.exists():
            file_path.unlink()
        
    except Exception:
        pass


def get_settings() -> Dict[str, Any]:
    """Get all application settings from persistent storage."""
    settings = get_cookie('app_settings', DEFAULT_SETTINGS.copy())
    
    # Ensure all default settings are present (for backward compatibility)
    for key, default_value in DEFAULT_SETTINGS.items():
        if key not in settings:
            settings[key] = default_value
    
    return settings


def get_setting(key: str, default: Any = None) -> Any:
    """Get a specific setting value."""
    settings = get_settings()
    return settings.get(key, default if default is not None else DEFAULT_SETTINGS.get(key))


def set_setting(key: str, value: Any) -> None:
    """Set a specific setting value and persist it."""
    settings = get_settings()
    settings[key] = value
    set_cookie('app_settings', settings, expires_days=365)  # Keep for 1 year
    
    # Also update session state for immediate access
    if 'app_settings' not in st.session_state:
        st.session_state.app_settings = {}
    st.session_state.app_settings[key] = value


def update_settings(new_settings: Dict[str, Any]) -> None:
    """Update multiple settings at once."""
    settings = get_settings()
    settings.update(new_settings)
    set_cookie('app_settings', settings, expires_days=365)
    
    # Also update session state
    if 'app_settings' not in st.session_state:
        st.session_state.app_settings = {}
    st.session_state.app_settings.update(new_settings)


def reset_settings() -> None:
    """Reset all settings to default values."""
    clear_cookie('app_settings')
    if 'app_settings' in st.session_state:
        del st.session_state.app_settings 