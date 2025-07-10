"""
Model Configuration Analysis Module

This module provides detailed analysis and visualization of HuggingFace model configurations.
"""

import streamlit as st
from typing import Dict, Any
from settings import get_setting


def format_number_with_units(value: int, use_units: bool = False) -> str:
    """
    Format a number with K, M, B, T units where K = 1024.
    
    Args:
        value: The number to format
        use_units: Whether to use unit formatting (K, M, B, T)
    
    Returns:
        Formatted string representation of the number
    """
    if not use_units or not isinstance(value, (int, float)):
        return f"{value:,}"
    
    # Convert to int if it's a float with no decimal part
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    
    # Define units with base 1024
    units = [
        (1024**4, 'T'),  # Tera
        (1024**3, 'B'),  # Billion (Giga)
        (1024**2, 'M'),  # Million (Mega)
        (1024**1, 'K'),  # Thousand (Kilo)
    ]
    
    for threshold, unit in units:
        if value >= threshold and value % threshold == 0:
            formatted_value = value // threshold
            return f"{formatted_value}{unit}"
    
    # If not a clean multiple of any unit, return with commas
    return f"{value:,}"


def display_model_statistics(config: Dict[str, Any]) -> None:
    """Display key model statistics in a metrics layout."""
    st.subheader("📊 Model Statistics")
    
    # Get the formatting preference from settings
    use_units = get_setting('use_number_units', False)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'num_hidden_layers' in config:
            st.metric("Layers", config['num_hidden_layers'])
    
    with col2:
        if 'hidden_size' in config:
            formatted_value = format_number_with_units(config['hidden_size'], use_units)
            st.metric("Hidden Size", formatted_value)
    
    with col3:
        if 'vocab_size' in config:
            formatted_value = format_number_with_units(config['vocab_size'], use_units)
            st.metric("Vocabulary", formatted_value)
    
    with col4:
        if 'num_attention_heads' in config:
            st.metric("Attention Heads", config['num_attention_heads'])


def calculate_parameter_estimation(config: Dict[str, Any]) -> None:
    """Calculate and display rough parameter estimation for the model."""
    required_keys = ['num_hidden_layers', 'hidden_size', 'vocab_size']
    
    if all(k in config for k in required_keys):
        # Get the formatting preference from settings
        use_units = get_setting('use_number_units', False)
        
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
        
        if use_units:
            # Try to format with units, fallback to millions if not a clean multiple
            formatted_params = format_number_with_units(total_params, use_units)
            if formatted_params == f"{total_params:,}":  # No unit was applied
                st.info(f"Estimated parameters: **{total_params/1e6:.1f}M** (rough calculation)")
            else:
                st.info(f"Estimated parameters: **{formatted_params}** (rough calculation)")
        else:
            st.info(f"Estimated parameters: **{total_params/1e6:.1f}M** (rough calculation)")
        
        st.caption("This is a simplified estimation and may not match the actual model size.")


def display_architecture_details(config: Dict[str, Any]) -> None:
    """Display detailed architecture information."""
    st.subheader("🏗️ Architecture Details")
    
    # Model type and architecture
    col1, col2 = st.columns(2)
    
    with col1:
        if 'model_type' in config:
            st.info(f"**Model Type:** {config['model_type']}")
        
        if 'architectures' in config:
            st.info(f"**Architectures:** {', '.join(config['architectures'])}")
    
    with col2:
        if 'torch_dtype' in config:
            st.info(f"**Data Type:** {config['torch_dtype']}")
        
        if 'transformers_version' in config:
            st.info(f"**Transformers Version:** {config['transformers_version']}")


def display_attention_details(config: Dict[str, Any]) -> None:
    """Display attention mechanism details."""
    attention_keys = [
        'num_attention_heads', 'num_key_value_heads', 'attention_dropout',
        'rope_theta', 'sliding_window'
    ]
    
    attention_config = {k: v for k, v in config.items() if k in attention_keys}
    
    if attention_config:
        st.subheader("🎯 Attention Configuration")
        
        # Get the formatting preference from settings
        use_units = get_setting('use_number_units', False)
        
        cols = st.columns(min(len(attention_config), 3))
        for i, (key, value) in enumerate(attention_config.items()):
            with cols[i % 3]:
                # Format key for display
                display_key = key.replace('_', ' ').title()
                
                # Format value if it's a number that could benefit from units
                if isinstance(value, (int, float)) and key in ['sliding_window', 'rope_theta']:
                    formatted_value = format_number_with_units(value, use_units)
                    st.metric(display_key, formatted_value)
                else:
                    st.metric(display_key, value)


def display_tokenization_info(config: Dict[str, Any]) -> None:
    """Display tokenization and vocabulary information."""
    token_keys = ['vocab_size', 'bos_token_id', 'eos_token_id', 'pad_token_id', 'unk_token_id']
    token_config = {k: v for k, v in config.items() if k in token_keys}
    
    if token_config:
        st.subheader("🔤 Tokenization")
        
        # Get the formatting preference from settings
        use_units = get_setting('use_number_units', False)
        
        # Vocabulary size as main metric
        if 'vocab_size' in token_config:
            formatted_vocab = format_number_with_units(token_config['vocab_size'], use_units)
            st.metric("Vocabulary Size", formatted_vocab)
        
        # Special tokens
        special_tokens = {k: v for k, v in token_config.items() if k != 'vocab_size'}
        if special_tokens:
            st.markdown("**Special Tokens:**")
            cols = st.columns(min(len(special_tokens), 4))
            for i, (key, value) in enumerate(special_tokens.items()):
                with cols[i % 4]:
                    display_key = key.replace('_token_id', '').replace('_', ' ').title()
                    st.text(f"{display_key}: {value}")


def display_training_details(config: Dict[str, Any]) -> None:
    """Display training and optimization details."""
    training_keys = [
        'hidden_dropout_prob', 'attention_dropout', 'layer_norm_eps',
        'initializer_range', 'rms_norm_eps', 'use_cache'
    ]
    
    training_config = {k: v for k, v in config.items() if k in training_keys}
    
    if training_config:
        st.subheader("⚙️ Training Configuration")
        
        # Group similar settings
        dropout_settings = {k: v for k, v in training_config.items() if 'dropout' in k}
        norm_settings = {k: v for k, v in training_config.items() if 'norm' in k or 'initializer' in k}
        other_settings = {k: v for k, v in training_config.items() 
                         if k not in dropout_settings and k not in norm_settings}
        
        if dropout_settings:
            st.markdown("**Dropout Settings:**")
            for key, value in dropout_settings.items():
                st.text(f"• {key.replace('_', ' ').title()}: {value}")
        
        if norm_settings:
            st.markdown("**Normalization & Initialization:**")
            for key, value in norm_settings.items():
                st.text(f"• {key.replace('_', ' ').title()}: {value}")
        
        if other_settings:
            st.markdown("**Other Settings:**")
            for key, value in other_settings.items():
                st.text(f"• {key.replace('_', ' ').title()}: {value}")


def display_full_analysis(config: Dict[str, Any]) -> None:
    """
    Main function to display comprehensive model configuration analysis.
    
    Args:
        config: The model configuration dictionary
    """
    st.markdown("### Configuration Analysis")
    
    # First row: Model Statistics (full width)
    display_model_statistics(config)
    
    st.markdown("---")
    
    # Second row: Architecture and Attention side by side
    col1, col2 = st.columns(2)
    with col1:
        display_architecture_details(config)
    with col2:
        display_attention_details(config)
    
    st.markdown("---")
    
    # Third row: Tokenization and Training side by side
    col1, col2 = st.columns(2)
    with col1:
        display_tokenization_info(config)
    with col2:
        display_training_details(config)
    
    st.markdown("---")
    
    # Fourth row: Parameter Estimation (full width)
    calculate_parameter_estimation(config) 