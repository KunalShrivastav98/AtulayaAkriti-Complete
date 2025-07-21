#!/usr/bin/env python3
"""
SAM Model Downloader for AtulayaAkriti
Downloads and sets up SAM model files automatically
"""

import os
import requests
from urllib.parse import urlparse
import sys
from pathlib import Path

def download_file(url, filename, chunk_size=8192):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        sys.stdout.write(f'\rDownloading {os.path.basename(filename)}: {progress:.1f}%')
                        sys.stdout.flush()

        print(f'\n✅ Downloaded {filename}')
        return True

    except Exception as e:
        print(f'\n❌ Error downloading {filename}: {e}')
        return False

def main():
    """Main download function"""
    print("🎨 AtulayaAkriti SAM Model Downloader")
    print("="*50)

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # SAM model options
    models = {
        "1": {
            "name": "ViT-B (Recommended)",
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "filename": "sam_vit_b_01ec64.pth",
            "size": "375MB"
        },
        "2": {
            "name": "ViT-L (Higher Quality)",
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "filename": "sam_vit_l_0b3195.pth",
            "size": "1.25GB"
        },
        "3": {
            "name": "ViT-H (Best Quality)",
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "filename": "sam_vit_h_4b8939.pth",
            "size": "2.56GB"
        }
    }

    print("Available SAM Models:")
    for key, model in models.items():
        print(f"  {key}. {model['name']} - {model['size']}")

    # Check for existing models
    existing_models = []
    for key, model in models.items():
        filepath = models_dir / model['filename']
        if filepath.exists():
            existing_models.append(key)
            print(f"  ✅ {model['name']} already exists")

    if existing_models:
        print(f"\n📋 Found {len(existing_models)} existing model(s)")
        choice = input("\nDo you want to download additional models? (y/n): ").strip().lower()
        if choice not in ['y', 'yes']:
            print("✅ Using existing models")
            return

    # Get user choice
    print("\n" + "="*50)
    choice = input("\nEnter model number to download (1-3) or 'all' for all models: ").strip()

    if choice.lower() == 'all':
        selected_models = models.keys()
    elif choice in models:
        selected_models = [choice]
    else:
        print("❌ Invalid choice")
        return

    # Download selected models
    print("\n🔄 Starting download(s)...")
    for model_key in selected_models:
        model = models[model_key]
        filepath = models_dir / model['filename']

        if filepath.exists():
            print(f"⏭️  Skipping {model['name']} (already exists)")
            continue

        print(f"\n📥 Downloading {model['name']} ({model['size']})...")
        success = download_file(model['url'], filepath)

        if not success:
            print(f"❌ Failed to download {model['name']}")
            return

    print("\n🎉 Download complete!")
    print("\n✅ Your AtulayaAkriti tool is now ready to use!")
    print("\nNext steps:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Run the app: streamlit run atulayakriti_app.py")

if __name__ == "__main__":
    main()
