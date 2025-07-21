#!/usr/bin/env python3
"""
Quick Start Script for AtulayaAkriti
Automated setup and launch
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and show progress"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False

def main():
    """Main quick start function"""
    print("🎨 AtulayaAkriti Quick Start")
    print("=" * 50)

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required")
        sys.exit(1)

    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

    # Create virtual environment
    if not os.path.exists("venv"):
        if not run_command(f"{sys.executable} -m venv venv", "Creating virtual environment"):
            sys.exit(1)

    # Determine activation command
    if platform.system() == "Windows":
        activate_cmd = "venv\\Scripts\\activate.bat"
        python_cmd = "venv\\Scripts\\python.exe"
    else:
        activate_cmd = "source venv/bin/activate"
        python_cmd = "venv/bin/python"

    # Install requirements
    install_cmd = f"{python_cmd} -m pip install -r requirements.txt"
    if not run_command(install_cmd, "Installing requirements"):
        sys.exit(1)

    # Download SAM model if needed
    model_files = [
        "models/sam_vit_b_01ec64.pth",
        "models/sam_vit_l_0b3195.pth",
        "models/sam_vit_h_4b8939.pth"
    ]

    if not any(os.path.exists(f) for f in model_files):
        print("📥 No SAM models found. Please run download_sam_model.py")
        choice = input("Download now? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            download_cmd = f"{python_cmd} download_sam_model.py"
            if not run_command(download_cmd, "Downloading SAM model"):
                sys.exit(1)

    # Test setup
    test_cmd = f"{python_cmd} test_setup.py"
    if not run_command(test_cmd, "Testing setup"):
        print("⚠️  Setup test failed, but continuing...")

    # Launch app
    print("\n🚀 Launching AtulayaAkriti...")
    launch_cmd = f"{python_cmd} -m streamlit run atulayakriti_app.py"

    try:
        subprocess.run(launch_cmd, shell=True)
    except KeyboardInterrupt:
        print("\n👋 AtulayaAkriti stopped")

if __name__ == "__main__":
    main()
