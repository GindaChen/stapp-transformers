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
from huggingface_hub import list_models
import re

# Page config
st.set_page_config(
    page_title="🤗 HuggingFace Model Config Viewer",
    page_icon="🤗",
    layout="wide",
    initial_sidebar_state="expanded"
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

# Create cache directory if it doesn't exist
CACHE_DIR.mkdir(exist_ok=True)

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


def set_cookie(key: str, value: Any, expires_days: int = 30) -> None:
    """Set a persistent value using local file storage."""
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
            # Session state not available (e.g., running outside Streamlit)
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
            # Session state not available
            pass


def get_cookie(key: str, default: Any = None) -> Any:
    """Get a persistent value from local file storage or session state."""
    try:
        # First check session state if available
        try:
            if 'persistent_data' in st.session_state and key in st.session_state.persistent_data:
                data = st.session_state.persistent_data[key]
                expires = datetime.fromisoformat(data['expires'])
                
                if datetime.now() < expires:
                    return data['value']
        except:
            # Session state not available
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
                    # Session state not available
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
    try:
        # Remove from session state if available
        try:
            if 'persistent_data' in st.session_state and key in st.session_state.persistent_data:
                del st.session_state.persistent_data[key]
        except:
            # Session state not available
            pass
        
        # Remove file
        user_data_dir = CACHE_DIR / "user_data"
        file_path = user_data_dir / f"{key}.json"
        if file_path.exists():
            file_path.unlink()
        
    except Exception:
        pass


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
    recent = recent[:MAX_RECENT_MODELS]
    
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
    expiry_time = cache_time + timedelta(hours=CACHE_EXPIRY_HOURS)
    
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
        
        if st.button("💾 Save Token"):
            set_hf_token(token_input)
            st.success("Token saved!")
        
        if current_token:
            st.success("✅ Token configured")
            if st.button("🗑️ Clear Token"):
                set_hf_token("")
                st.success("Token cleared!")
                st.rerun()
        
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
                hf_suggestions = search_huggingface_models(manual_input, limit=8)
            
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
        
        st.markdown("---")
        
        # Recent Models
        st.subheader("🕒 Recently Viewed")
        recent_models = get_recent_models()
        if recent_models:
            for model in recent_models[:10]:
                if st.button(f"📄 {model}", key=f"recent_{model}", use_container_width=True):
                    st.session_state.selected_model = model
                    st.rerun()
        else:
            st.info("No recent models")
    
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
            
            # Display success message
            cache_status = "🟢 Cached" if result['cached'] else "🔵 Fresh"
            st.success(f"Loaded configuration for **{result['model_name']}** {cache_status}")
            
        except Exception as e:
            st.error(f"❌ Error loading config: {str(e)}")
            st.session_state.last_result = None
    
    # Display results if available
    if hasattr(st.session_state, 'last_result') and st.session_state.last_result:
        result = st.session_state.last_result
        
        # Header with model info
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"{result['model_name']}")
        
        with col2:
            model_name = result['model_name']
            hf_url = f"https://huggingface.co/{model_name}"
            st.markdown(f"[Go to Model Page]({hf_url})", unsafe_allow_html=True)
        
        

        
        # Tabs for different views
        tab1, tab2 = st.tabs(["🗂️ Raw JSON", "📖 Full Analysis"])
        
        with tab1:
            st.markdown("### Raw Configuration JSON")
            
            # # Display config size and structure info
            # col1, col2, col3 = st.columns(3)
            # with col1:
            #     st.metric("Total Keys", len(result['config']))
            # with col2:
            #     # Count nested objects and arrays
            #     nested_count = sum(1 for v in result['config'].values() if isinstance(v, (dict, list)))
            #     st.metric("Nested Objects", nested_count)
            # with col3:
            #     config_json = json.dumps(result['config'], indent=2)
            #     st.metric("JSON Size", f"{len(config_json):,} chars")
            
            # st.markdown("---")
            
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
            st.markdown("### Configuration Analysis")
            
            config = result['config']
            
            # Model statistics
            st.subheader("📊 Model Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'num_hidden_layers' in config:
                    st.metric("Layers", config['num_hidden_layers'])
            
            with col2:
                if 'hidden_size' in config:
                    st.metric("Hidden Size", f"{config['hidden_size']:,}")
            
            with col3:
                if 'vocab_size' in config:
                    st.metric("Vocabulary", f"{config['vocab_size']:,}")
            
            with col4:
                if 'num_attention_heads' in config:
                    st.metric("Attention Heads", config['num_attention_heads'])
            
            # Calculate model size estimation
            if all(k in config for k in ['num_hidden_layers', 'hidden_size', 'vocab_size']):
                # Rough parameter estimation
                hidden_size = config['hidden_size']
                num_layers = config['num_hidden_layers']
                vocab_size = config['vocab_size']
                
                # Embedding parameters
                embedding_params = vocab_size * hidden_size
                
                # Transformer layer parameters (simplified)
                attention_params = 4 * hidden_size * hidden_size  # Q, K, V, O projections
                ffn_params = 2 * hidden_size * config.get('intermediate_size', 4 * hidden_size)
                layer_params = attention_params + ffn_params
                total_layer_params = num_layers * layer_params
                
                # Total parameters (rough estimate)
                total_params = embedding_params + total_layer_params
                
                st.subheader("🧮 Parameter Estimation")
                st.info(f"Estimated parameters: **{total_params/1e6:.1f}M** (rough calculation)")
                st.caption("This is a simplified estimation and may not match the actual model size.")
    
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