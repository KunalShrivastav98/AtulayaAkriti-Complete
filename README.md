# AtulayaAkriti Texture Rendering Tool

**Version:** 1.0.0  
**Author:** Your Name  
**License:** MIT

## Project Overview
The AtulayaAkriti Texture Rendering Tool is a professional-grade web application for photorealistic interior design visualization. Built with Streamlit and Meta's Segment Anything Model (SAM), it allows users to upload interior images, select regions via point-and-click, and apply realistic textures or solid colors to those regions.

![AtulayaAkriti Banner](docs/banner.png)

## Key Features
- **AI-Powered Segmentation:** Utilizes SAM for precise region selection.
- **Advanced Texture Blending:** Edge-aware feathering, LAB color matching, seamless tiling.
- **Interactive UI:** Modern, responsive interface with real-time feedback.
- **Easy Setup:** One-click scripts for Windows, macOS, and Linux.
- **Download Results:** Save processed images in high resolution.

## Directory Structure
```text
AtulayaAkriti-Complete/
├── atulayakriti_app.py          # Main Streamlit app
├── requirements.txt             # Python dependencies
├── download_sam_model.py        # SAM model downloader
├── test_setup.py                # Installation test script
├── run_app.bat                  # Windows launcher
├── run_app.sh                   # Linux/Mac launcher
├── quick_start.py               # Cross-platform quick start script
├── models/                      # Place SAM model files here
├── assets/
│   └── textures/                # Texture images
├── utils/                       # Utility scripts (future expansion)
├── docs/                        # Documentation and images
└── README.md                    # Project documentation
```

## Quick Start
### Windows
```bash
run_app.bat
```
### macOS/Linux
```bash
chmod +x run_app.sh
./run_app.sh
```
### Python Script (Any OS)
```bash
python quick_start.py
```

## Manual Setup
1. **Clone the Repository**
   ```bash
   git clone https://your-repo-url.git
   cd AtulayaAkriti-Complete
   ```
2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scriptsctivate
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download SAM Model**
   ```bash
   python download_sam_model.py
   ```
5. **Run the Application**
   ```bash
   streamlit run atulayakriti_app.py
   ```

## Troubleshooting
- **Missing Dependencies:** Run `pip install -r requirements.txt`.
- **SAM Library Error:** Install SAM with `pip install git+https://github.com/facebookresearch/segment-anything.git`.
- **No SAM Model Files:** Run `python download_sam_model.py` to download the ViT-B model.

## Contributing
Contributions are welcome! Please open issues and pull requests.

## License
This project is licensed under the MIT License. See `LICENSE` for details.
