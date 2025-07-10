> ![WARNING]
> GPT Generated README - not yet reviewed by human!

# 🤗 HuggingFace Model Config Viewer


A simple Streamlit web application to explore and understand HuggingFace model configurations with detailed explanations and caching.

## Features

- 🔍 **Search any HuggingFace model** with autocomplete and suggestions
- 🔑 **HuggingFace token support** for private and gated models
- 🕒 **Recently viewed models** with persistent session storage
- 📚 **Model categories** automatically organized from registry
- 💾 **Local caching** for faster subsequent loads (24-hour expiry)
- 🎨 **Modern Streamlit UI** with sidebar navigation and clean design
- 📈 **Model analysis** with parameter estimation and statistics
- 💾 **Download configs** as JSON files
- 🗂️ **Two view modes** - raw JSON and full analysis (not implemented yet)
- 📝 **Model registry** system - simple list of model names for autocomplete

## Quick Start

### Option 1: Automatic Setup (Recommended)
```bash
cd transformers
./run.sh
```

### Option 2: Manual Setup
```bash
cd transformers

# Install dependencies
pip install -r requirements.txt

# Start the Streamlit app
streamlit run app.py
```

The app will automatically open in your browser, typically at: **http://localhost:8501**

## Usage

### Getting Started
1. **Configure HuggingFace Token** (optional) - Add your token in the sidebar for private models
2. **Search for models** - Use Quick Select for autocomplete or Manual Entry for any model
3. **Browse categories** - Explore models organized by type (Chat, Base, Task-Specific, etc.)
4. **View recent models** - Quickly access your recently viewed models

### Model Views
- **🗂️ Raw JSON**: Complete configuration in JSON format with download option
- **📖 Full Analysis**: Model statistics, parameter estimation, and key metrics

### Features
- **Automatic categorization** - Models are grouped by type based on naming patterns
- **Recent history** - Last 10 viewed models are saved in your session
- **Token management** - Securely store your HuggingFace token for private models
- **Cache management** - View cache status and clear cached configs

### Example Models to Try

- `Qwen/Qwen2.5-7B-Instruct` - Qwen large language model
- `microsoft/DialoGPT-medium` - Conversational AI model
- `google/flan-t5-large` - Google's T5 model for instruction following
- `meta-llama/Llama-2-7b-hf` - Meta's Llama 2 model
- `mistralai/Mistral-7B-v0.1` - Mistral's efficient language model

## How It Works

### Streamlit Application

- **Fetches configs** directly from HuggingFace Hub using their public API
- **Caches responses** locally in `config_cache/` directory (24-hour expiry)
- **Handles errors** gracefully (model not found, private models, etc.)
- **Organized display** with parameters grouped by category (Basic, Architecture, etc.)
- **Interactive explanations** for 25+ common configuration parameters
- **Multiple view modes** for different use cases
- **Real-time cache management** with sidebar controls

### Caching System

- Configs are cached in `config_cache/` directory
- Cache files are named using MD5 hash of model name for safety
- 24-hour expiry to ensure configs stay reasonably fresh
- Cache status is shown in the UI (Cached/Fresh)

## Application Structure

The Streamlit app provides an intuitive interface with several sections:

### Main Interface
- **Model Input**: Text field to enter HuggingFace model names
- **Load Button**: Fetch and display configuration
- **Three View Tabs**: Different ways to explore the config data

### Sidebar Features
- **HuggingFace Token**: Secure token storage for private/gated models
- **Model Search**: Two modes - Quick Select (autocomplete) and Manual Entry (with live suggestions)
- **Recent Models**: Quick access to last 10 viewed models
- **Model Categories**: Auto-categorized models from registry
  - 💬 Chat & Instruct (instruction-tuned models)
  - 🧠 Base Models (foundation models)
  - 🎯 Task-Specific (specialized models)
  - 🔬 Research (experimental models)
  - 📚 Other (miscellaneous models)
- **Cache Management**: View cache status and clear cached files

### Configuration Views
1. **Raw JSON**: Complete config with syntax highlighting and download
2. **Full Analysis**: Statistics, parameter estimation, and model insights

## Configuration Parameters Explained

The viewer provides explanations for common configuration parameters:

- **`model_type`** - The architecture type (e.g., "gpt2", "bert", "llama")
- **`vocab_size`** - Number of unique tokens the model can process
- **`hidden_size`** - Dimension of hidden representations/embeddings
- **`num_hidden_layers`** - Number of transformer layers
- **`num_attention_heads`** - Number of attention heads per layer
- **`max_position_embeddings`** - Maximum sequence length
- And many more...

## File Structure

```
transformers/
├── app.py               # Main Streamlit application
├── model_registry.json  # Model registry (list of model names for autocomplete)
├── requirements.txt     # Python dependencies
├── run.sh              # Startup script
├── README.md           # This file
└── config_cache/       # Cache directory (created automatically)
```

## Dependencies

- **Python 3.7+**
- **Streamlit 1.29.0** - Web application framework
- **requests 2.31.0** - HTTP library for API calls

## Error Handling

The application handles various error scenarios:

- **Model not found** (404) - Shows clear error message
- **Private/gated models** (403) - Explains access restrictions
- **Network issues** - Provides helpful error messages
- **Invalid model names** - Input validation and suggestions
- **Server errors** - Graceful degradation with error details

## Cache Management

- Configs are automatically cached for 24 hours
- Cache files are stored in `config_cache/` directory
- Use `/api/cache/info` to see what's cached
- Use `/api/cache/clear` to clear all cached files
- Cache status is shown in the UI for each loaded config

## Security

- Input validation on model names
- Safe filename generation using MD5 hashing
- CORS protection for API endpoints
- No arbitrary code execution
- Read-only access to HuggingFace Hub

## Performance

- Cached configs load instantly
- Streaming JSON parser for large configurations
- Optimized frontend rendering
- Minimal dependencies for fast startup

## Troubleshooting

### App won't start
- Check that Python 3.7+ is installed
- Ensure Streamlit is installed: `pip install streamlit`
- Try running manually: `streamlit run app.py`

### Model not loading
- Verify the model name is correct
- Check if the model exists on HuggingFace Hub
- Some models may be private or gated

### Cache issues
- Use the "🗑️ Clear Cache" button in the sidebar
- Or manually delete cache directory: `rm -rf config_cache/`

## Contributing

This is a standalone educational tool. Feel free to modify and extend it for your needs!

## License

This project is open source and available under the MIT License. 