#!/usr/bin/env python3
"""
HuggingFace Model Config Viewer - Streamlit App

A Streamlit application that fetches and displays model configurations 
from HuggingFace Hub with detailed explanations and caching.
"""

import streamlit as st
import requests
import json
import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import time
from huggingface_hub import list_models, HfApi
import re

# Page config
st.set_page_config(
    page_title="HuggingFace Model Config Viewer",
    page_icon="🤗",
    layout="wide",
    initial_sidebar_state="expanded"  # Note: This will be managed by settings in future versions
)

# Configuration
CACHE_DIR = Path("config_cache")
CACHE_EXPIRY_HOURS = 24
HF_CONFIG_URL = "https://huggingface.co/{model_name}/resolve/main/config.json"
MODEL_REGISTRY_FILE = Path("model_registry.json")
if not MODEL_REGISTRY_FILE.exists():
    with open(MODEL_REGISTRY_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f)
MAX_RECENT_MODELS = 10

# Default settings
DEFAULT_SETTINGS = {
    'use_number_units': False,
    'cache_expiry_hours': 24,
    'max_recent_models': 10,
    'auto_refresh_on_settings_change': True,
    'sidebar_expanded': True
}

# Create cache directory if it doesn't exist
CACHE_DIR.mkdir(exist_ok=True)

# Initialize settings module
from settings import set_cache_dir, get_setting, set_setting, update_settings, reset_settings, get_cookie, set_cookie, clear_cookie
set_cache_dir(CACHE_DIR)

# Initialize safetensors cache
import safetensors_info
safetensors_info.set_safetensors_cache_dir(CACHE_DIR)

# Configuration explanations for common parameters
CONFIG_EXPLANATIONS = {
    'model_type': 'The type/architecture of the model (e.g., "gpt2", "bert", "llama")',
    'vocab_size': 'Size of the vocabulary - number of unique tokens the model can process',
    'hidden_size': 'Dimension of the hidden representations/embeddings',
    'num_hidden_layers': 'Number of transformer layers in the model',
    'num_attention_heads': 'Number of attention heads in each layer',
    'intermediate_size': 'Dimension of the feed-forward network hidden layer',
    'max_position_embeddings': 'Maximum sequence length the model can handle',
    'architectures': 'List of model architectures this config is compatible with',
    'torch_dtype': 'PyTorch data type used for model weights',
    'transformers_version': 'Version of transformers library used to create this config',
    'use_cache': 'Whether to cache key/value pairs during generation for efficiency',
    'attention_dropout': 'Dropout probability applied to attention weights',
    'hidden_dropout_prob': 'Dropout probability applied to hidden layers',
    'layer_norm_eps': 'Epsilon value for layer normalization',
    'initializer_range': 'Standard deviation for weight initialization',
    'rope_theta': 'Base frequency for RoPE (Rotary Position Embedding)',
    'sliding_window': 'Size of the sliding attention window (for models like Mistral)',
    'tie_word_embeddings': 'Whether input and output embeddings share weights',
    'rms_norm_eps': 'Epsilon value for RMS normalization',
    'num_key_value_heads': 'Number of key-value heads for grouped-query attention',
    'pretraining_tp': 'Tensor parallelism degree used during pretraining',
    'bos_token_id': 'ID of the beginning-of-sequence token',
    'eos_token_id': 'ID of the end-of-sequence token',
    'pad_token_id': 'ID of the padding token',
    'unk_token_id': 'ID of the unknown token'
}


def get_cache_path(model_name: str) -> Path:
    """Generate cache file path for a model."""
    safe_name = hashlib.md5(model_name.encode()).hexdigest()
    return CACHE_DIR / f"{safe_name}.json"


def load_model_registry() -> List[str]:
    """Load model registry from file."""
    if MODEL_REGISTRY_FILE.exists():
        try:
            with open(MODEL_REGISTRY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle both old dict format and new list format
                if isinstance(data, dict):
                    return list(data.keys())
                return data
        except (json.JSONDecodeError, IOError):
            pass
    return []


def save_model_registry(registry: List[str]) -> None:
    """Save model registry to file."""
    try:
        with open(MODEL_REGISTRY_FILE, 'w', encoding='utf-8') as f:
            json.dump(sorted(list(set(registry))), f, indent=2, ensure_ascii=False)
    except IOError:
        pass


def add_model_to_registry(model_name: str) -> None:
    """Add a model to the registry."""
    registry = load_model_registry()
    if model_name not in registry:
        registry.append(model_name)
        save_model_registry(registry)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def search_huggingface_models(query: str, limit: int = 10) -> List[str]:
    """Search HuggingFace Hub for models matching the query."""
    if not query or len(query) < 2:
        return []
    
    try:
        # Search models on HuggingFace Hub
        models = list_models(
            search=query,
            sort="downloads",
            direction=-1,
            limit=limit
        )
        
        # Extract model names (id field)
        model_names = []
        for model in models:
            if hasattr(model, 'id') and model.id:
                model_names.append(model.id)
        
        return model_names[:limit]
        
    except Exception as e:
        # If HuggingFace search fails, return empty list
        return []


# Settings functions are now in settings.py module


def render_settings_menu() -> Dict[str, Any]:
    """
    Render the settings menu in the sidebar and return any changed settings.
    
    Returns:
        Dictionary of changed settings (empty if no changes)
    """
    st.subheader("🎨 Settings")
    
    # Get current settings
    from settings import get_settings
    current_settings = get_settings()
    changed_settings = {}
    
    # Display Settings Section
    with st.expander("📊 Display Settings", expanded=True):
        # Number formatting toggle
        use_units = st.checkbox(
            "Use K/M/B/T units for numbers",
            value=current_settings['use_number_units'],
            help="Format numbers like 4096 as 4K (where K = 1024)",
            key="setting_use_units"
        )
        
        if use_units != current_settings['use_number_units']:
            changed_settings['use_number_units'] = use_units
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox(
            "Auto-refresh on settings change",
            value=current_settings['auto_refresh_on_settings_change'],
            help="Automatically refresh the display when settings change",
            key="setting_auto_refresh"
        )
        
        if auto_refresh != current_settings['auto_refresh_on_settings_change']:
            changed_settings['auto_refresh_on_settings_change'] = auto_refresh
    
    # Cache Settings Section
    with st.expander("💾 Cache Settings", expanded=False):
        cache_hours = st.slider(
            "Cache expiry (hours)",
            min_value=1,
            max_value=168,  # 1 week
            value=current_settings['cache_expiry_hours'],
            help="How long to keep cached model configurations",
            key="setting_cache_hours"
        )
        
        if cache_hours != current_settings['cache_expiry_hours']:
            changed_settings['cache_expiry_hours'] = cache_hours
        
        max_recent = st.slider(
            "Max recent models",
            min_value=5,
            max_value=50,
            value=current_settings['max_recent_models'],
            help="Maximum number of recent models to remember",
            key="setting_max_recent"
        )
        
        if max_recent != current_settings['max_recent_models']:
            changed_settings['max_recent_models'] = max_recent
        
        # Cache management buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Cache", help="Clear all cached model configurations and safetensors data"):
                # Clear all cache files
                if CACHE_DIR.exists():
                    for cache_file in CACHE_DIR.glob("*.json"):
                        if cache_file.name != "model_registry.json":
                            cache_file.unlink()
                    
                    # Clear safetensors cache
                    safetensors_cache_dir = CACHE_DIR / "safetensors"
                    if safetensors_cache_dir.exists():
                        for cache_file in safetensors_cache_dir.glob("*.json"):
                            cache_file.unlink()
                
                st.success("Cache cleared!")
        
        with col2:
            if st.button("🔄 Reset Settings", help="Reset all settings to defaults"):
                reset_settings()
                st.success("Settings reset!")
                st.rerun()
    
    # Advanced Settings Section
    with st.expander("⚙️ Advanced Settings", expanded=False):
        sidebar_expanded = st.checkbox(
            "Sidebar expanded by default",
            value=current_settings['sidebar_expanded'],
            help="Whether the sidebar should be expanded when the app loads",
            key="setting_sidebar_expanded"
        )
        
        if sidebar_expanded != current_settings['sidebar_expanded']:
            changed_settings['sidebar_expanded'] = sidebar_expanded
        
        # Export/Import settings
        st.markdown("**Settings Management:**")
        
        # Export settings
        settings_json = json.dumps(current_settings, indent=2)
        st.download_button(
            "📥 Export Settings",
            data=settings_json,
            file_name="hf_config_viewer_settings.json",
            mime="application/json",
            help="Download current settings as JSON file"
        )
        
        # Import settings
        uploaded_file = st.file_uploader(
            "📤 Import Settings",
            type=['json'],
            help="Upload a settings JSON file to restore settings",
            key="settings_upload"
        )
        
        if uploaded_file is not None:
            try:
                imported_settings = json.load(uploaded_file)
                # Validate imported settings
                valid_settings = {}
                for key, value in imported_settings.items():
                    if key in DEFAULT_SETTINGS:
                        valid_settings[key] = value
                
                if valid_settings:
                    update_settings(valid_settings)
                    st.success(f"Imported {len(valid_settings)} settings!")
                    st.rerun()
                else:
                    st.error("No valid settings found in the uploaded file.")
            except json.JSONDecodeError:
                st.error("Invalid JSON file.")
            except Exception as e:
                st.error(f"Error importing settings: {str(e)}")
    
    # Apply changed settings
    if changed_settings:
        update_settings(changed_settings)
        
        # Auto-refresh if enabled
        if current_settings.get('auto_refresh_on_settings_change', True):
            if hasattr(st.session_state, 'last_result') and st.session_state.last_result:
                st.rerun()
    
    return changed_settings


def get_recent_models() -> List[str]:
    """Get recently viewed models from cookies or session state."""
    try:
        # Try to get from cookies first
        recent_models = get_cookie('recent_models', [])
        
        # If not in cookies, check session state
        if not recent_models and 'recent_models' in st.session_state:
            recent_models = st.session_state.recent_models
        
        # Ensure it's a list
        if not isinstance(recent_models, list):
            recent_models = []
        
        # Store in session state for current session
        st.session_state.recent_models = recent_models
        
        return recent_models
    except:
        # Fallback for testing outside Streamlit
        return []


def add_recent_model(model_name: str) -> None:
    """Add a model to recent models list and save to cookies."""
    recent = get_recent_models()
    if model_name in recent:
        recent.remove(model_name)
    recent.insert(0, model_name)
    
    # Use setting for max recent models
    max_recent = get_setting('max_recent_models', MAX_RECENT_MODELS)
    recent = recent[:max_recent]
    
    # Update both session state and cookies
    st.session_state.recent_models = recent
    set_cookie('recent_models', recent, expires_days=90)  # Keep for 3 months


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from cookies or session state."""
    try:
        # Try to get from cookies first
        token = get_cookie('hf_token', None)
        
        # If not in cookies, check session state
        if not token and 'hf_token' in st.session_state:
            token = st.session_state.hf_token
        
        # Store in session state for current session
        if token:
            st.session_state.hf_token = token
        
        return token
    except:
        return None


def set_hf_token(token: str) -> None:
    """Set HuggingFace token in session state and cookies."""
    clean_token = token.strip() if token else None
    
    # Update both session state and cookies
    st.session_state.hf_token = clean_token
    
    if clean_token:
        set_cookie('hf_token', clean_token, expires_days=365)  # Keep for 1 year
    else:
        clear_cookie('hf_token')


def is_cache_valid(cache_path: Path) -> bool:
    """Check if cached file exists and is not expired."""
    if not cache_path.exists():
        return False
    
    cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
    # Use setting for cache expiry
    cache_hours = get_setting('cache_expiry_hours', CACHE_EXPIRY_HOURS)
    expiry_time = cache_time + timedelta(hours=cache_hours)
    
    return datetime.now() < expiry_time


def load_from_cache(cache_path: Path) -> Dict[str, Any]:
    """Load config from cache file."""
    with open(cache_path, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)
    
    return {
        'config': cache_data['config'],
        'model_name': cache_data['model_name'],
        'cached': True,
        'cache_time': cache_data['cache_time'],
        'cached_at': cache_data.get('cached_at', 'Unknown')
    }


def save_to_cache(cache_path: Path, model_name: str, config: Dict[str, Any]) -> None:
    """Save config to cache file."""
    cache_data = {
        'model_name': model_name,
        'config': config,
        'cache_time': datetime.now().isoformat(),
        'cached_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    }
    
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)


def fetch_config_from_hf(model_name: str, token: Optional[str] = None) -> Dict[str, Any]:
    """Fetch model config from HuggingFace Hub."""
    url = HF_CONFIG_URL.format(model_name=model_name)
    
    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        config = response.json()
        return config
        
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            if e.response.status_code == 404:
                raise ValueError(f"Model '{model_name}' not found on HuggingFace Hub")
            elif e.response.status_code == 401:
                raise ValueError(f"Unauthorized access to model '{model_name}'. Please check your HuggingFace token.")
            elif e.response.status_code == 403:
                raise ValueError(f"Access denied for model '{model_name}'. It might be private or gated.")
        raise ValueError(f"Failed to fetch config: {str(e)}")


def validate_model_name(model_name: str) -> str:
    """Validate and clean model name."""
    if not model_name or not model_name.strip():
        raise ValueError("Model name cannot be empty")
    
    model_name = model_name.strip()
    return model_name


def get_model_config(model_name: str, token: Optional[str] = None) -> Dict[str, Any]:
    """Get model configuration with caching."""
    model_name = validate_model_name(model_name)
    cache_path = get_cache_path(model_name)
    
    # Try to load from cache first
    if is_cache_valid(cache_path):
        return load_from_cache(cache_path)
    
    # Fetch from HuggingFace Hub
    config = fetch_config_from_hf(model_name, token)
    
    # Save to cache and registry
    save_to_cache(cache_path, model_name, config)
    add_model_to_registry(model_name)
    
    return {
        'config': config,
        'model_name': model_name,
        'cached': False,
        'cache_time': datetime.now().isoformat(),
        'cached_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    }


def get_model_categories() -> Dict[str, List[str]]:
    """Get model categories from registry."""
    registry = load_model_registry()
    
    # Group models by common patterns
    categories = {
        "💬 Chat & Instruct": [],
        "🧠 Base Models": [],
        "🎯 Task-Specific": [],
        "🔬 Research": [],
        "📚 Other": []
    }
    
    for model_name in registry:
        model_lower = model_name.lower()
        if any(keyword in model_lower for keyword in ['chat', 'instruct', 'dialog']):
            categories["💬 Chat & Instruct"].append(model_name)
        elif any(keyword in model_lower for keyword in ['base', 'foundation', 'llama', 'mistral', 'qwen']):
            categories["🧠 Base Models"].append(model_name)
        elif any(keyword in model_lower for keyword in ['bert', 'roberta', 'sentence', 'embed']):
            categories["🎯 Task-Specific"].append(model_name)
        elif any(keyword in model_lower for keyword in ['research', 'experimental', 'alpha', 'beta']):
            categories["🔬 Research"].append(model_name)
        else:
            categories["📚 Other"].append(model_name)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🤗 HuggingFace Model Config Viewer")
    st.markdown("Explore and understand model configurations with detailed explanations")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings")
        
        # HuggingFace Token
        st.subheader("🔑 HuggingFace Token")
        current_token = get_hf_token()
        token_input = st.text_input(
            "Token (optional)",
            value=current_token or "",
            type="password",
            help="Enter your HuggingFace token for private/gated models",
            placeholder="hf_..."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Token"):
                set_hf_token(token_input)
                st.success("Token saved!")
        with col2:
            if current_token and st.button("🗑️ Clear Token"):
                set_hf_token("")
                st.success("Token cleared!")
                st.rerun()
        
        if current_token:
            st.success("✅ Token configured")
        
        
        
        st.markdown("---")
        
        # Model Search
        st.subheader("🔍 Model Search")
        
        # Get registry for autocomplete
        registry = load_model_registry()
        
        # Combined search - both dropdown and text input
        with st.form("model_search_form"):
            # Searchable selectbox that allows both selection and manual entry
            selected_model = st.selectbox(
                "Select or search for a model",
                options=[""] + registry,
                index=0,
                help="Start typing to search through available models, or select from the list"
            )
            
            # Manual entry field for models not in registry
            manual_input = st.text_input(
                "Or enter model name manually",
                value="",
                placeholder="e.g., microsoft/DialoGPT-medium",
                help="Enter any model name from HuggingFace Hub (press Enter to load)"
            )
            
            # Form submit button
            form_submitted = st.form_submit_button("📥 Load Config", type="primary", use_container_width=True)
            
            # Determine which model to use
            model_name = manual_input if manual_input else selected_model
            
            if form_submitted and model_name:
                st.session_state.selected_model = model_name
                st.rerun()
            elif form_submitted and not model_name:
                st.error("Please select or enter a model name")
        
        # Show HuggingFace suggestions for manual input (outside form to avoid conflicts)
        if manual_input and len(manual_input) >= 3:  # Require at least 3 characters for better search
            with st.spinner("Searching HuggingFace Hub..."):
                hf_suggestions = search_huggingface_models(manual_input, limit=20)
            
            if hf_suggestions:
                st.markdown("**🤗 HuggingFace Suggestions:**")
                st.caption(f"Found {len(hf_suggestions)} models matching '{manual_input}'")
                
                # Display suggestions in a single column
                for suggestion in hf_suggestions:
                    if st.button(f"📋 {suggestion}", key=f"hf_suggest_{suggestion}", use_container_width=True):
                        st.session_state.selected_model = suggestion
                        st.rerun()
            else:
                st.info(f"💡 No models found on HuggingFace Hub matching '{manual_input}'. Try a different search term.")
        
        # Show current selection
        if model_name:
            st.info(f"Selected: {model_name}")
        
        load_config = False  # Form submission handles loading
        
        # st.markdown("---")
        
        st.subheader("🕒 Recently Viewed")
        recent_models = get_recent_models()
        
        if recent_models:
            # Initialize page state
            if 'recent_page' not in st.session_state:
                st.session_state.recent_page = 0
            
            models_per_page = 3
            total_pages = (len(recent_models) + models_per_page - 1) // models_per_page
            current_page = st.session_state.recent_page
            
            # Ensure current page is valid
            if current_page >= total_pages:
                st.session_state.recent_page = 0
                current_page = 0
            
            # Get models for current page
            start_idx = current_page * models_per_page
            end_idx = min(start_idx + models_per_page, len(recent_models))
            page_models = recent_models[start_idx:end_idx]
            
            # Display models for current page
            for model in page_models:
                if st.button(f"📄 {model}", key=f"recent_{model}_{current_page}", use_container_width=True):
                    st.session_state.selected_model = model
                    st.rerun()
            
            # Pagination controls
            if total_pages > 1:
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    if st.button("◀ Prev", disabled=(current_page == 0), use_container_width=True):
                        st.session_state.recent_page = current_page - 1
                        st.rerun()
                
                with col2:
                    st.markdown(f"<div style='text-align: center; padding: 8px;'>Page {current_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
                
                with col3:
                    if st.button("Next ▶", disabled=(current_page == total_pages - 1), use_container_width=True):
                        st.session_state.recent_page = current_page + 1
                        st.rerun()
        else:
            st.info("No recent models")

        st.markdown("---")
        
        # Render settings menu
        render_settings_menu()
    
    # Main content area
    # Check if we need to load a model
    selected_model = st.session_state.get('selected_model', None)
    
    # Don't clear selected_model if it was set by recently viewed buttons
    # Only clear if user explicitly cleared the search form
    
    # Load and display config
    if selected_model:
        try:
            with st.spinner(f"Loading configuration for {selected_model}..."):
                result = get_model_config(selected_model, get_hf_token())
            
            # Add to recent models
            add_recent_model(selected_model)
            
            # Store in session state
            st.session_state.last_result = result
            
        except Exception as e:
            st.error(f"❌ Error loading config: {str(e)}")
            st.session_state.last_result = None
    
    # Display results if available
    if hasattr(st.session_state, 'last_result') and st.session_state.last_result:
        result = st.session_state.last_result
        
        # Header with model info
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"## {result['model_name']}")
        
        with col2:
            model_name = result['model_name']
            hf_url = f"https://huggingface.co/{model_name}"
            st.markdown(f"[Go to Model Page]({hf_url})", unsafe_allow_html=True)
        
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["🗂️ Raw JSON", "📖 Full Analysis", "🔒 Safetensors"])
        
        with tab1:            
            # Format JSON with proper indentation and make it copyable
            formatted_json = json.dumps(result['config'], indent=2, ensure_ascii=False)
            
            # Display JSON in a code block for better readability
            st.code(formatted_json, language='json', line_numbers=True)
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download button
                st.download_button(
                    label="💾 Download Config JSON",
                    data=formatted_json,
                    file_name=f"{result['model_name'].replace('/', '_')}_config.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                # Copy to clipboard button (using a text area for easy copying)
                if st.button("📋 Copy to Clipboard", use_container_width=True):
                    st.session_state.show_copy_text = True
            
            with col3:
                # Pretty print button to show minified vs formatted
                if 'show_minified' not in st.session_state:
                    st.session_state.show_minified = False
                
                if st.button("🗜️ Toggle Minified", use_container_width=True):
                    st.session_state.show_minified = not st.session_state.show_minified
                    st.rerun()
            
            # Show copy text area if requested
            if st.session_state.get('show_copy_text', False):
                st.text_area(
                    "Select all and copy (Ctrl+A, Ctrl+C):",
                    value=formatted_json,
                    height=100,
                    help="Select all text and copy to clipboard"
                )
                if st.button("❌ Hide Copy Area"):
                    st.session_state.show_copy_text = False
                    st.rerun()
            
            # Show minified version if toggled
            if st.session_state.get('show_minified', False):
                st.markdown("**Minified JSON:**")
                minified_json = json.dumps(result['config'], separators=(',', ':'))
                st.code(minified_json, language='json')
            
            # Show key explorer
            st.markdown("---")
            st.markdown("**🔍 Key Explorer**")
            
            # Create a searchable list of all keys
            all_keys = []
            
            def extract_keys(obj, prefix=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        full_key = f"{prefix}.{key}" if prefix else key
                        all_keys.append((full_key, type(value).__name__, str(value)[:100] if not isinstance(value, (dict, list)) else f"{type(value).__name__} with {len(value)} items"))
                        if isinstance(value, (dict, list)):
                            extract_keys(value, full_key)
                elif isinstance(obj, list) and len(obj) > 0:
                    for i, item in enumerate(obj[:3]):  # Show first 3 items
                        full_key = f"{prefix}[{i}]" if prefix else f"[{i}]"
                        all_keys.append((full_key, type(item).__name__, str(item)[:100] if not isinstance(item, (dict, list)) else f"{type(item).__name__}"))
            
            extract_keys(result['config'])
            
            # Filter keys
            if all_keys:
                search_key = st.text_input("🔍 Search keys:", placeholder="Type to filter keys...")
                
                filtered_keys = all_keys
                if search_key:
                    filtered_keys = [k for k in all_keys if search_key.lower() in k[0].lower()]
                
                if filtered_keys:
                    # Display keys in a nice table format
                    st.markdown(f"**Found {len(filtered_keys)} keys:**")
                    
                    # Create a dataframe for better display
                    import pandas as pd
                    df = pd.DataFrame(filtered_keys, columns=['Key Path', 'Type', 'Value Preview'])
                    st.dataframe(df, use_container_width=True, height=min(400, len(filtered_keys) * 35 + 50))
        
        with tab2:
            import analysis
            analysis.display_full_analysis(result['config'])
        
        with tab3:
            import safetensors_info
            safetensors_info.display_safetensors_page(result['model_name'], get_hf_token())
    
    else:
        # Show welcome message
        st.markdown("## Welcome! 👋")
        st.markdown("""
        Use the sidebar to:
        - 🔍 **Search for models** by entering a model name
        - 🕒 **View recently loaded models** for quick access
        - 📚 **Browse model categories** to discover new models
        - 🔑 **Configure your HuggingFace token** for private models
        
        Start by entering a model name in the sidebar search box!
        """)


if __name__ == "__main__":
    main() 