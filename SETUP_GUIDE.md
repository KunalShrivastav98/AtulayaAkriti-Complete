# AtulayaAkriti Setup Guide

Follow these steps to install and run the AtulayaAkriti Texture Rendering Tool.

## 1. Requirements
- Python 3.8 or higher
- pip (Python package manager)
- 4GB RAM (8GB recommended)
- Stable internet connection (for model download)

## 2. Clone or Download Project
If you have git:
```bash
git clone https://your-repo-url.git
cd AtulayaAkriti-Complete
```
Or download the ZIP and extract it.

## 3. Create and Activate Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scriptsctivate
# macOS/Linux
source venv/bin/activate
```

## 4. Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Download SAM Model
SAM requires a pre-trained model file. Run the downloader:
```bash
python download_sam_model.py
```
Choose option 1 (ViT-B) for the quickest download. The file will be saved in `models/`.

## 6. Run Setup Tests (Optional)
Verify everything is installed correctly:
```bash
python test_setup.py
```
You should see all tests pass.

## 7. Launch the Application
### Windows
```bash
run_app.bat
```
### macOS/Linux
```bash
chmod +x run_app.sh
./run_app.sh
```
Alternatively, run with Python directly:
```bash
streamlit run atulayakriti_app.py
```

## 8. Using the App
1. Upload an interior image.
2. Click on the region you want to texture.
3. Load the SAM model (if not already loaded).
4. Choose texture and blending options.
5. Click "Apply Texture".
6. Download the result.

## 9. Updating Dependencies
Update dependencies in `requirements.txt` and reinstall:
```bash
pip install -r requirements.txt --upgrade
```

## 10. Uninstalling
Just delete the project folder. Virtual environment is self-contained.

---
Enjoy using AtulayaAkriti! For issues, open a GitHub ticket.
