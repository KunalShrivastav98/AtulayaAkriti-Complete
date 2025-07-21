#!/bin/bash
echo "====================================================="
echo "  🎨 AtulayaAkriti Texture Rendering Tool"
echo "  Professional Interior Design Visualization"
echo "====================================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "✅ Python found"
echo

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔄 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "🔄 Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "🔄 Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "❌ requirements.txt not found"
    exit 1
fi

# Check if SAM model exists
echo "🔄 Checking for SAM models..."
if [ ! -f "models/sam_vit_b_01ec64.pth" ] && [ ! -f "models/sam_vit_l_0b3195.pth" ] && [ ! -f "models/sam_vit_h_4b8939.pth" ]; then
    echo "📥 No SAM models found. Starting download..."
    python download_sam_model.py
fi

# Test setup
echo "🔄 Testing setup..."
python test_setup.py

# Launch the application
echo
echo "🚀 Launching AtulayaAkriti..."
echo
echo "Your browser will open automatically."
echo "Press Ctrl+C to stop the application."
echo
streamlit run atulayakriti_app.py
