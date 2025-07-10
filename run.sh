#!/bin/bash

# HuggingFace Model Config Viewer - Startup Script

echo "🤗 Starting HuggingFace Model Config Viewer..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi


# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create cache directory
echo "📂 Creating cache directory..."
mkdir -p config_cache

# Start the Streamlit app
echo "🚀 Starting Streamlit app..."
echo "📖 Your browser should open automatically to the app"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

streamlit run app.py 