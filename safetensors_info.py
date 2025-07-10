"""
Safetensors Information Module

This module fetches and displays safetensors metadata from HuggingFace models.
"""

import streamlit as st
import json
import requests
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import hashlib
from datetime import datetime, timedelta

# Cache configuration
SAFETENSORS_CACHE_DIR = None  # Will be set by set_cache_dir

def set_safetensors_cache_dir(cache_dir: Path) -> None:
    """Set the cache directory for safetensors data."""
    global SAFETENSORS_CACHE_DIR
    SAFETENSORS_CACHE_DIR = cache_dir / "safetensors"
    SAFETENSORS_CACHE_DIR.mkdir(exist_ok=True)

def get_safetensors_cache_path(model_name: str) -> Path:
    """Generate cache file path for safetensors data."""
    if SAFETENSORS_CACHE_DIR is None:
        raise ValueError("Cache directory not set. Call set_safetensors_cache_dir first.")
    
    safe_name = hashlib.md5(model_name.encode()).hexdigest()
    return SAFETENSORS_CACHE_DIR / f"safetensors_{safe_name}.json"

def is_safetensors_cache_valid(cache_path: Path, cache_expiry_hours: int = 24) -> bool:
    """Check if cached safetensors file exists and is not expired."""
    if not cache_path.exists():
        return False
    
    cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
    expiry_time = cache_time + timedelta(hours=cache_expiry_hours)
    
    return datetime.now() < expiry_time

def load_safetensors_from_cache(cache_path: Path) -> Dict[str, Any]:
    """Load safetensors data from cache file."""
    with open(cache_path, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)
    
    return {
        **cache_data['safetensors_info'],
        'cached': True,
        'cache_time': cache_data['cache_time'],
        'cached_at': cache_data.get('cached_at', 'Unknown')
    }

def save_safetensors_to_cache(cache_path: Path, model_name: str, safetensors_info: Dict[str, Any]) -> None:
    """Save safetensors data to cache file."""
    cache_data = {
        'model_name': model_name,
        'safetensors_info': safetensors_info,
        'cache_time': datetime.now().isoformat(),
        'cached_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    }
    
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)


def fetch_safetensors_index(model_name: str, token: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch safetensors index file directly from HuggingFace Hub.
    
    Args:
        model_name: Name of the model
        token: Optional HuggingFace token
    
    Returns:
        Dictionary containing index data with tensor metadata
    """
    url = f"https://huggingface.co/{model_name}/resolve/main/model.safetensors.index.json"
    
    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        index_data = response.json()
        return index_data
        
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            if e.response.status_code == 404:
                raise ValueError(f"Safetensors index file not found for model '{model_name}'")
            elif e.response.status_code == 401:
                raise ValueError(f"Unauthorized access to model '{model_name}'. Please check your HuggingFace token.")
            elif e.response.status_code == 403:
                raise ValueError(f"Access denied for model '{model_name}'. It might be private or gated.")
        raise ValueError(f"Failed to fetch safetensors index: {str(e)}")


def fetch_single_safetensors_file(model_name: str, token: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch a single safetensors file directly to extract tensor metadata.
    
    Args:
        model_name: Name of the model
        token: Optional HuggingFace token
    
    Returns:
        Dictionary containing tensor metadata
    """
    url = f"https://huggingface.co/{model_name}/resolve/main/model.safetensors"
    
    headers = {'Range': 'bytes=0-8191'}  # Get first 8KB to read header
    if token:
        headers['Authorization'] = f'Bearer {token}'
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse safetensors header
        data = response.content
        if len(data) < 8:
            return {}
        
        # Read header length (first 8 bytes)
        header_length = int.from_bytes(data[:8], 'little')
        
        # Make sure we have enough data
        if len(data) < 8 + header_length:
            # Need to fetch more data
            headers['Range'] = f'bytes=0-{8 + header_length - 1}'
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.content
        
        # Extract header JSON
        header_bytes = data[8:8 + header_length]
        header_json = json.loads(header_bytes.decode('utf-8'))
        
        return header_json
        
    except Exception as e:
        raise ValueError(f"Failed to fetch safetensors metadata: {str(e)}")


def fetch_tensor_metadata_from_file(model_name: str, file_path: str, token: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch tensor metadata from a specific safetensors file by reading just the header.
    
    Args:
        model_name: Name of the model
        file_path: Path to the safetensors file in the repo
        token: Optional HuggingFace token
    
    Returns:
        Dictionary containing tensor metadata (shapes, dtypes)
    """
    url = f"https://huggingface.co/{model_name}/resolve/main/{file_path}"
    
    try:
        # First, get header length (first 8 bytes)
        headers = {'Range': 'bytes=0-7'}
        if token:
            headers['Authorization'] = f'Bearer {token}'
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        if len(response.content) < 8:
            return {}
        
        header_length = int.from_bytes(response.content, 'little')
        
        # Now fetch the full header
        headers = {'Range': f'bytes=0-{8 + header_length - 1}'}
        if token:
            headers['Authorization'] = f'Bearer {token}'
            
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.content
        if len(data) < 8 + header_length:
            return {}
        
        # Extract header JSON
        header_bytes = data[8:8 + header_length]
        header_json = json.loads(header_bytes.decode('utf-8'))
        
        # Extract just the tensor metadata (exclude __metadata__ section)
        tensor_metadata = {}
        for key, value in header_json.items():
            if key != "__metadata__" and isinstance(value, dict):
                if "shape" in value or "dtype" in value:
                    tensor_metadata[key] = value
        
        return tensor_metadata
        
    except Exception as e:
        st.warning(f"Could not fetch metadata from {file_path}: {str(e)}")
        return {}


def get_safetensors_info(model_name: str, token: Optional[str] = None, cache_expiry_hours: int = 24) -> Dict[str, Any]:
    """
    Get safetensors information for a model with caching support.
    
    Args:
        model_name: Name of the model on HuggingFace Hub
        token: Optional HuggingFace token for private models
        cache_expiry_hours: Cache expiry time in hours
    
    Returns:
        Dictionary containing safetensors information
    """
    # Check cache first
    try:
        cache_path = get_safetensors_cache_path(model_name)
        if is_safetensors_cache_valid(cache_path, cache_expiry_hours):
            return load_safetensors_from_cache(cache_path)
    except (ValueError, Exception):
        # Cache not available or not set up, continue without caching
        pass
    
    result = {
        "model_name": model_name,
        "has_safetensors": False,
        "has_index": False,
        "cached": False
    }
    
    # First, try to fetch the index file
    try:
        index_data = fetch_safetensors_index(model_name, token)
        result["has_safetensors"] = True
        result["has_index"] = True
        result["index_data"] = index_data
        
        # Extract tensor metadata from index
        weight_map = index_data.get("weight_map", {})
        metadata = index_data.get("metadata", {})
        
        # Now fetch tensor metadata from actual safetensors files
        all_tensor_metadata = {}
        unique_files = list(set(weight_map.values()))
        
        # Fetch metadata from each unique file
        for file_path in unique_files:
            file_metadata = fetch_tensor_metadata_from_file(model_name, file_path, token)
            all_tensor_metadata.update(file_metadata)
        
        # Combine with weight mapping
        tensor_metadata = {}
        for tensor_name, file_name in weight_map.items():
            tensor_info = {"file": file_name}
            
            # Add shape and dtype if available
            if tensor_name in all_tensor_metadata:
                tensor_info.update(all_tensor_metadata[tensor_name])
            
            tensor_metadata[tensor_name] = tensor_info
        
        result["tensor_metadata"] = tensor_metadata
        
    except ValueError as e:
        # Index file not found, try to fetch single safetensors file
        try:
            single_file_metadata = fetch_single_safetensors_file(model_name, token)
            if single_file_metadata:
                result["has_safetensors"] = True
                result["has_index"] = False
                
                # Process single file metadata
                tensor_metadata = {}
                for key, value in single_file_metadata.items():
                    if key != "__metadata__" and isinstance(value, dict):
                        if "shape" in value or "dtype" in value:
                            tensor_metadata[key] = {
                                "file": "model.safetensors",
                                **value
                            }
                
                result["tensor_metadata"] = tensor_metadata
                result["note"] = "Model uses single safetensors file (no index)"
            else:
                result["error"] = f"No safetensors files found for model '{model_name}'"
        except ValueError:
            result["error"] = f"Model '{model_name}' does not use safetensors format"
    
    # Save to cache if successful and cache is available
    try:
        if not result.get("error") and SAFETENSORS_CACHE_DIR is not None:
            cache_path = get_safetensors_cache_path(model_name)
            save_safetensors_to_cache(cache_path, model_name, result)
    except Exception:
        # Cache save failed, but don't fail the whole operation
        pass
    
    return result


def format_dtype(dtype_str: str) -> str:
    """Format dtype string for display."""
    dtype_mapping = {
        'F32': 'float32',
        'F16': 'float16', 
        'BF16': 'bfloat16',
        'I64': 'int64',
        'I32': 'int32',
        'I16': 'int16',
        'I8': 'int8',
        'U8': 'uint8',
        'BOOL': 'bool'
    }
    return dtype_mapping.get(dtype_str, dtype_str)





def display_tensor_table(weight_map: Dict[str, str], tensor_metadata: Dict[str, Any] = None) -> None:
    """
    Display tensors in a table format maintaining original order.
    
    Args:
        weight_map: Weight mapping from index file
        tensor_metadata: Optional tensor metadata with shapes and dtypes
    """
    st.markdown("#### 🗂️ Tensor Information")
    
    # Filter bar
    filter_text = st.text_input(
        "🔍 Filter tensors",
        placeholder="Type to filter by tensor name...",
        help="Filter tensors by name (case-insensitive)"
    )
    
    # Prepare table data
    table_data = []
    ordered_tensors = list(weight_map.keys())  # Maintain original order
    
    def calculate_tensor_size_bytes(shape, dtype_str):
        """Calculate tensor size in bytes."""
        if not shape:
            return 0
        
        # Calculate total elements
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        
        # Bytes per element based on dtype
        dtype_bytes = {
            'float32': 4, 'F32': 4,
            'float16': 2, 'F16': 2,
            'bfloat16': 2, 'BF16': 2,
            'int64': 8, 'I64': 8,
            'int32': 4, 'I32': 4,
            'int16': 2, 'I16': 2,
            'int8': 1, 'I8': 1,
            'uint8': 1, 'U8': 1,
            'bool': 1, 'BOOL': 1
        }
        
        bytes_per_element = dtype_bytes.get(dtype_str, 4)  # Default to 4 bytes
        return total_elements * bytes_per_element
    
    def format_size_bytes(size_bytes):
        """Format bytes into human-readable size."""
        if size_bytes >= 1_000_000_000:  # GB
            return f"{size_bytes / 1_000_000_000:.1f} GB"
        elif size_bytes >= 1_000_000:  # MB
            return f"{size_bytes / 1_000_000:.1f} MB"
        elif size_bytes >= 1_000:  # KB
            return f"{size_bytes / 1_000:.1f} KB"
        else:
            return f"{size_bytes} B"
    
    for tensor_name in ordered_tensors:
        # Get tensor metadata if available
        shape_str = "Unknown"
        dtype_str = "Unknown"
        elements_str = "Unknown"
        size_str = "Unknown"
        
        if tensor_metadata and tensor_name in tensor_metadata:
            metadata = tensor_metadata[tensor_name]
            
            if 'shape' in metadata:
                shape = metadata['shape']
                shape_str = " × ".join(map(str, shape))
                
                # Calculate total elements
                total_elements = 1
                for dim in shape:
                    total_elements *= dim
                
                if total_elements >= 1_000_000_000:
                    elements_str = f"{total_elements/1_000_000_000:.1f}B"
                elif total_elements >= 1_000_000:
                    elements_str = f"{total_elements/1_000_000:.1f}M"
                elif total_elements >= 1_000:
                    elements_str = f"{total_elements/1_000:.1f}K"
                else:
                    elements_str = str(total_elements)
            
            if 'dtype' in metadata:
                original_dtype = metadata['dtype']
                dtype_str = format_dtype(original_dtype)
                
                # Calculate size if we have shape
                if 'shape' in metadata:
                    size_bytes = calculate_tensor_size_bytes(metadata['shape'], original_dtype)
                    size_str = format_size_bytes(size_bytes)
        
        table_data.append({
            "Tensor Name": tensor_name,
            "Shape": shape_str,
            "Data Type": dtype_str,
            "Elements": elements_str,
            "Size": size_str
        })
    
    # Filter data if filter text is provided
    if filter_text:
        filtered_data = [
            row for row in table_data 
            if filter_text.lower() in row["Tensor Name"].lower()
        ]
    else:
        filtered_data = table_data
    
    # Display as dataframe
    if filtered_data:
        import pandas as pd
        df = pd.DataFrame(filtered_data)
        
        # Display with search and pagination
        st.dataframe(
            df,
            use_container_width=True,
            height=min(600, len(filtered_data) * 35 + 50),  # Dynamic height with max
            hide_index=True
        )
        
        # Show summary
        if filter_text:
            st.markdown(f"**Showing {len(filtered_data)} of {len(table_data)} tensors** (filtered)")
        else:
            st.markdown(f"**Total tensors:** {len(table_data)}")
            
    elif filter_text:
        st.info(f"No tensors found matching '{filter_text}'")
    else:
        st.info("No tensor information available.")





def display_metadata_cascade(metadata: Dict[str, Any]) -> None:
    """Display metadata in a cascade view."""
    st.markdown("#### 📋 Metadata Cascade")
    
    def display_metadata_item(key: str, value: Any, level: int = 0) -> None:
        indent = "  " * level
        
        if isinstance(value, dict):
            st.markdown(f"{indent}📂 **{key}**")
            for sub_key, sub_value in value.items():
                display_metadata_item(sub_key, sub_value, level + 1)
        elif isinstance(value, list):
            if len(value) <= 5:
                st.markdown(f"{indent}📄 **{key}**: {value}")
            else:
                st.markdown(f"{indent}📄 **{key}**: [{', '.join(map(str, value[:3]))}, ... +{len(value)-3} more]")
        else:
            if isinstance(value, str) and len(value) > 100:
                st.markdown(f"{indent}📄 **{key}**: {value[:100]}...")
            else:
                st.markdown(f"{indent}📄 **{key}**: {value}")
    
    for key, value in metadata.items():
        display_metadata_item(key, value)


def display_safetensors_info(safetensors_info: Dict[str, Any]) -> None:
    """
    Display safetensors information in Streamlit.
    
    Args:
        safetensors_info: Safetensors information dictionary
    """
    st.markdown("### 🔒 Safetensors Information")
    
    if "error" in safetensors_info:
        st.error(f"❌ {safetensors_info['error']}")
        return
    
    model_name = safetensors_info.get("model_name", "Unknown")
    
    # Basic information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        has_safetensors = safetensors_info.get("has_safetensors", False)
        st.metric("Uses Safetensors", "✅ Yes" if has_safetensors else "❌ No")
    
    with col2:
        safetensors_files = safetensors_info.get("safetensors_files", [])
        st.metric("Safetensors Files", len(safetensors_files))
    
    with col3:
        index_files = safetensors_info.get("index_files", [])
        st.metric("Index Files", len(index_files))
    
    if not has_safetensors:
        st.warning("This model does not use Safetensors format.")
        return
    
    with col4:
        if safetensors_info.get("has_index"):
            st.markdown(f"[📋 Index file](https://huggingface.co/{model_name}?show_file_info=model.safetensors.index.json)")
    
    # Show cache status
    if safetensors_info.get("cached"):
        st.info(f"ℹ️ Data loaded from cache (cached at: {safetensors_info.get('cached_at', 'unknown time')})")
    
    # Display index information if available
    if "index_data" in safetensors_info:
        index_data = safetensors_info["index_data"]
        
        # Metadata cascade view
        metadata = index_data.get("metadata", {})
        if metadata:
            display_metadata_cascade(metadata)
            st.markdown("---")
        
        # Weight map with tensor metadata
        weight_map = index_data.get("weight_map", {})
        tensor_metadata = safetensors_info.get("tensor_metadata", {})
        
        if weight_map:
            display_tensor_table(weight_map, tensor_metadata)
        
        if False:
            st.markdown("---")
            
            # Show file summary
            st.markdown("#### 📄 File Summary")
            unique_files = sorted(set(weight_map.values()))
            
            for i, file_name in enumerate(unique_files):
                tensors_in_file = [k for k, v in weight_map.items() if v == file_name]
                st.text(f"{i+1}. {file_name} ({len(tensors_in_file)} tensors)")
    
    elif "index_error" in safetensors_info:
        st.warning(f"⚠️ Could not load index file: {safetensors_info['index_error']}")
    
    elif safetensors_info.get("note"):
        st.info(f"ℹ️ {safetensors_info['note']}")
        
        # Try to show tensor metadata if available
        tensor_metadata = safetensors_info.get("tensor_metadata", {})
        if tensor_metadata:
            st.markdown("#### 🗂️ Available Tensors")
            for tensor_name, metadata in tensor_metadata.items():
                if tensor_name != '__metadata__':
                    shape_str = ""
                    dtype_str = ""
                    
                    if 'shape' in metadata:
                        shape_str = format_tensor_size(metadata['shape'])
                    
                    if 'dtype' in metadata:
                        dtype_str = format_dtype(metadata['dtype'])
                    
                    info_parts = [p for p in [shape_str, dtype_str] if p]
                    info_str = " • ".join(info_parts)
                    
                    st.markdown(f"• **{tensor_name}** → {info_str}")
    
    # Show direct safetensors files if any
    safetensors_files = safetensors_info.get("safetensors_files", [])
    if safetensors_files:
        st.markdown("#### 📋 Safetensors Files")
        for i, file_name in enumerate(safetensors_files):
            st.text(f"{i+1}. {file_name}")


def display_safetensors_page(model_name: str, token: Optional[str] = None) -> None:
    """
    Display the complete safetensors information page.
    
    Args:
        model_name: Name of the model
        token: Optional HuggingFace token
    """
    if not model_name:
        st.info("No model loaded. Please load a model configuration first.")
        return
    
    # Get cache expiry from settings
    try:
        from settings import get_setting
        cache_expiry_hours = get_setting('cache_expiry_hours', 24)
    except ImportError:
        cache_expiry_hours = 24
    
    with st.spinner(f"Fetching safetensors information for {model_name}..."):
        safetensors_info = get_safetensors_info(model_name, token, cache_expiry_hours)
    
    display_safetensors_info(safetensors_info) 