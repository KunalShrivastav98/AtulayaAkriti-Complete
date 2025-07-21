import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import cv2
import io
import base64
import os
import requests
import json
from typing import Optional, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import torch with error handling for Streamlit Cloud
try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PyTorch not available: {e}")
    TORCH_AVAILABLE = False

from streamlit_drawable_canvas import st_canvas

# Set page config first
st.set_page_config(
    page_title="AtulayaAkriti Texture Rendering Tool",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E7D32 0%, #FFA726 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E7D32;
        margin: 1rem 0;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
    }
    .warning-message {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
    }
    .stButton > button {
        background: linear-gradient(90deg, #2E7D32 0%, #FFA726 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

class SAMIntegration:
    """SAM (Segment Anything Model) Integration Class"""

    def __init__(self):
        self.sam_model = None
        self.sam_predictor = None
        self.model_loaded = False
        self.available_models = self.check_available_models()

    def check_available_models(self) -> List[str]:
        """Check for available SAM model files and verify they're not corrupted"""
        model_files = [
            "sam_vit_b_01ec64.pth",
            "sam_vit_l_0b3195.pth", 
            "sam_vit_h_4b8939.pth",
            "sam_vit_b.pth"  # Legacy name
        ]

        available = []
        for model_file in model_files:
            model_path = f"models/{model_file}"
            if os.path.exists(model_path):
                # Verify the model file is not corrupted
                if self.verify_model_file(model_path):
                    available.append(model_file)
                else:
                    logger.warning(f"Corrupted model file detected: {model_file}")
                    # Remove corrupted file
                    try:
                        os.remove(model_path)
                        logger.info(f"Removed corrupted file: {model_file}")
                    except Exception as e:
                        logger.error(f"Failed to remove corrupted file {model_file}: {e}")

        return available

    def verify_model_file(self, model_path: str) -> bool:
        """Verify that the SAM model file is valid and not corrupted"""
        try:
            if not os.path.exists(model_path):
                return False
            
            # Check file size (should be around 375MB for ViT-B)
            file_size = os.path.getsize(model_path)
            if file_size < 300_000_000:  # Less than 300MB indicates incomplete download
                logger.warning(f"Model file incomplete: {file_size / (1024*1024):.1f}MB")
                return False
            
            # Special check for Git LFS pointer files (common on Streamlit Cloud)
            with open(model_path, 'rb') as f:
                first_bytes = f.read(100)
                if b'version https://git-lfs.github.com/spec/v1' in first_bytes:
                    logger.warning("Detected Git LFS pointer file instead of actual model")
                    return False
            
            # Try to load the model to verify it's not corrupted
            if TORCH_AVAILABLE:
                try:
                    # Use map_location='cpu' to avoid GPU issues on Streamlit Cloud
                    model_data = torch.load(model_path, map_location='cpu', weights_only=True)
                    # Basic validation - SAM models should have specific keys
                    if not isinstance(model_data, dict):
                        logger.warning("Model file doesn't contain expected dictionary structure")
                        return False
                    return True
                except Exception as e:
                    logger.warning(f"Model file corrupted or invalid: {e}")
                    return False
            else:
                # If torch is not available, just check file size and format
                logger.info("PyTorch not available, skipping model validation")
                return True
                
        except Exception as e:
            logger.warning(f"Error verifying model: {e}")
            return False

    def auto_download_model(self) -> bool:
        """Auto-download ViT-B model if no models are available or corrupted"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            
            model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            model_filename = "models/sam_vit_b_01ec64.pth"
            
            # Check if file exists and is valid
            if os.path.exists(model_filename):
                if self.verify_model_file(model_filename):
                    st.success("✅ Valid SAM model found!")
                    self.available_models = self.check_available_models()
                    return True
                else:
                    # Remove corrupted file
                    try:
                        os.remove(model_filename)
                        st.info("🗑️ Removed corrupted model file")
                    except:
                        pass
            
            st.info("🔄 Downloading SAM ViT-B model (375MB) - this may take a few minutes...")
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Download with progress tracking
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(model_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = downloaded / total_size
                            progress_bar.progress(progress)
                            status_text.text(f"Downloaded: {downloaded // (1024*1024):.1f}MB / {total_size // (1024*1024):.1f}MB")
            
            progress_bar.progress(1.0)
            status_text.text("✅ Download complete!")
            st.success("✅ SAM ViT-B model downloaded successfully!")
            
            # Refresh available models
            self.available_models = self.check_available_models()
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            st.info("💡 You can manually download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
            return False

    def load_sam_model(self, model_path: str = None) -> bool:
        """Load SAM model with error handling"""
        try:
            if not TORCH_AVAILABLE:
                st.error("❌ PyTorch is not available. Please check your installation.")
                return False
                
            # Try to import SAM
            try:
                from segment_anything import sam_model_registry, SamPredictor
                st.success("✅ SAM library imported successfully!")
            except ImportError as e:
                st.error(f"❌ SAM library not found: {e}")
                st.info("Please install SAM with: pip install git+https://github.com/facebookresearch/segment-anything.git")
                return False

            # Determine model path
            if model_path is None:
                if self.available_models:
                    model_path = f"models/{self.available_models[0]}"
                else:
                    st.error("❌ No SAM model files found!")
                    return False

            # Verify model file before loading
            if not self.verify_model_file(model_path):
                st.error(f"❌ Model file {model_path} is invalid or corrupted!")
                st.info("💡 Try downloading a fresh model file")
                return False

            # Determine model type
            if "vit_b" in model_path:
                model_type = "vit_b"
            elif "vit_l" in model_path:
                model_type = "vit_l"
            elif "vit_h" in model_path:
                model_type = "vit_h"
            else:
                model_type = "vit_b"  # Default

            # Load model with proper error handling
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            with st.spinner(f"Loading {model_type} model..."):
                self.sam_model = sam_model_registry[model_type](checkpoint=model_path)
                self.sam_model.to(device=device)
                self.sam_predictor = SamPredictor(self.sam_model)
                self.model_loaded = True

            st.success(f"✅ SAM model loaded successfully! ({model_type} on {device})")
            return True

        except Exception as e:
            st.error(f"❌ Error loading SAM model: {e}")
            logger.error(f"SAM model loading error: {e}")
            return False

    def generate_masks(self, image: np.ndarray, points: List[Tuple[int, int]]):
        """Generate all segmentation masks and scores from points"""
        if not self.model_loaded:
            st.error("❌ SAM model not loaded!")
            return None, None
        try:
            self.sam_predictor.set_image(image)
            input_points = np.array(points)
            input_labels = np.ones(len(points))
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True
            )
            return masks, scores
        except Exception as e:
            st.error(f"❌ Error generating masks: {e}")
            logger.error(f"Mask generation error: {e}")
            return None, None

class TextureApplicator:
    """Advanced texture application with realistic blending"""

    @staticmethod
    def create_solid_color_texture(color: Tuple[int, int, int], size: Tuple[int, int]) -> Image.Image:
        """Create a solid color texture"""
        return Image.new('RGB', size, color)

    @staticmethod
    def tile_texture_to_image(texture: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Tile the texture image to cover the target size naturally (no stretching)"""
        tiled = Image.new('RGB', target_size)
        tex_w, tex_h = texture.size
        for x in range(0, target_size[0], tex_w):
            for y in range(0, target_size[1], tex_h):
                tiled.paste(texture, (x, y))
        return tiled

    @staticmethod
    def apply_texture_to_mask(
        original_image: Image.Image,
        mask: np.ndarray,
        texture: Image.Image,
        blend_mode: str = "normal",
        opacity: float = 1.0,
        is_tiled: bool = False
    ) -> Image.Image:
        """Apply texture to masked area with advanced blending"""
        try:
            # Always ensure both images are RGB (3 channels)
            original_np = np.array(original_image.convert("RGB"))
            if is_tiled:
                texture_resized = TextureApplicator.tile_texture_to_image(texture.convert("RGB"), original_image.size)
            else:
                texture_resized = texture.convert("RGB").resize(original_image.size, Image.LANCZOS)
            texture_np = np.array(texture_resized)
            # Improved edge feathering
            mask_np = mask.astype(np.uint8) * 255
            mask_smooth = cv2.GaussianBlur(mask_np, (31, 31), 0)  # Larger blur for softer edge
            mask_smooth = mask_smooth / 255.0
            mask_smooth *= opacity
            if blend_mode == "photorealistic_lab":
                orig_lab = cv2.cvtColor(original_np, cv2.COLOR_RGB2LAB)
                text_lab = cv2.cvtColor(texture_np, cv2.COLOR_RGB2LAB)
                blended_lab = orig_lab.copy()
                for c in [1, 2]:
                    blended_lab[..., c] = (
                        orig_lab[..., c] * (1 - mask_smooth) + text_lab[..., c] * mask_smooth
                    ).astype(np.uint8)
                result = cv2.cvtColor(blended_lab, cv2.COLOR_LAB2RGB)
                result = np.clip(result, 0, 255).astype(np.uint8)
                return Image.fromarray(result)
            elif blend_mode == "normal":
                result = original_np * (1 - mask_smooth[..., np.newaxis]) + texture_np * mask_smooth[..., np.newaxis]
            elif blend_mode == "multiply":
                blended = (original_np * texture_np) / 255.0
                result = original_np * (1 - mask_smooth[..., np.newaxis]) + blended * mask_smooth[..., np.newaxis]
            elif blend_mode == "overlay":
                overlay = np.where(original_np < 128, 
                                 2 * original_np * texture_np / 255.0,
                                 255 - 2 * (255 - original_np) * (255 - texture_np) / 255.0)
                result = original_np * (1 - mask_smooth[..., np.newaxis]) + overlay * mask_smooth[..., np.newaxis]
            else:
                result = original_np * (1 - mask_smooth[..., np.newaxis]) + texture_np * mask_smooth[..., np.newaxis]
            result = np.clip(result, 0, 255).astype(np.uint8)
            return Image.fromarray(result)
        except Exception as e:
            st.error(f"❌ Error applying texture: {e}")
            logger.error(f"Texture application error: {e}")
            return original_image

def get_image_download_link(img: Image.Image, filename: str) -> str:
    """Generate download link for image"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}" class="download-button">📥 Download {filename}</a>'
    return href

def main():
    """Main application function"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎨 AtulayaAkriti Texture Rendering Tool</h1>
        <p>Professional AI-Powered Interior Design Texture Application</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize SAM
    if 'sam_integration' not in st.session_state:
        st.session_state.sam_integration = SAMIntegration()
        
        # Force auto-download on Streamlit Cloud (models directory will be empty)
        if not st.session_state.sam_integration.available_models:
            st.info("🚀 First-time setup on Streamlit Cloud - downloading SAM model...")
            with st.spinner("Downloading SAM model (375MB) - this may take 2-3 minutes..."):
                if st.session_state.sam_integration.auto_download_model():
                    st.success("✅ Model downloaded successfully! Refreshing app...")
                    st.rerun()  # Refresh the app after download
                else:
                    st.error("❌ Failed to download model. Please try refreshing the page.")

    # Sidebar
    with st.sidebar:
        st.header("🛠️ Controls")

        # Model Status
        st.subheader("📊 Model Status")
        if st.session_state.sam_integration.available_models:
            st.success(f"✅ {len(st.session_state.sam_integration.available_models)} SAM model(s) found")
            selected_model = st.selectbox(
                "Select SAM Model:",
                st.session_state.sam_integration.available_models,
                help="Choose the SAM model to use for segmentation"
            )

            if st.button("🔄 Load SAM Model"):
                model_path = f"models/{selected_model}"
                st.session_state.sam_integration.load_sam_model(model_path)
        else:
            st.warning("⚠️ No SAM models found!")
            
            # Auto-download option
            if st.button("🚀 Auto-Download ViT-B Model (375MB)", help="Automatically download the recommended model"):
                if st.session_state.sam_integration.auto_download_model():
                    st.rerun()
            
            st.markdown("**Or manually download:**")
            st.markdown("""
            <div class="warning-message">
                <strong>Manual Download Options:</strong><br>
                • <a href="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" target="_blank">ViT-B (375MB)</a><br>
                • <a href="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth" target="_blank">ViT-L (1.25GB)</a><br>
                • <a href="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" target="_blank">ViT-H (2.56GB)</a><br>
                <br>
                <small>Place downloaded files in the <code>models/</code> directory.</small>
            </div>
            """, unsafe_allow_html=True)

        # Texture Options
        st.subheader("🎨 Texture Options")
        texture_type = st.selectbox(
            "Texture Type:",
            ["Solid Color", "Pattern Texture", "Wood Texture", "Upload Texture/Color"],
            help="Choose the type of texture to apply"
        )

        import os
        from PIL import Image as PILImage

        texture = None
        texture_name = None
        upload_color_rgb = (255, 255, 255)
        upload_texture_uploaded = False
        if texture_type == "Solid Color":
            color_option = st.selectbox(
                "Color Preset:",
                ["Custom", "Greens & Blues", "Yellows & Earth Tones", "Pinks & Reds", "Purples & Browns"],
                help="Choose from professional color categories"
            )

            if color_option == "Custom":
                selected_color = st.color_picker("Pick a color:", "#FFFFFF")
                color_rgb = tuple(int(selected_color[i:i+2], 16) for i in (1, 3, 5))
            elif color_option == "Greens & Blues":
                green_color = st.selectbox(
                    "Select Green/Blue:",
                    ["Saga Green (22153 MR+)", "Lotto Green (22133 MR+)", "Jolly Green (21103 MR+)", 
                     "Tropical Blue (22062 MR+)", "Parrot Green (21143 MR+)"],
                    help="Professional green and blue tones"
                )
                green_colors = {
                    "Saga Green (22153 MR+)": (180, 190, 170),
                    "Lotto Green (22133 MR+)": (160, 170, 140),
                    "Jolly Green (21103 MR+)": (140, 160, 120),
                    "Tropical Blue (22062 MR+)": (120, 160, 180),
                    "Parrot Green (21143 MR+)": (100, 180, 120)
                }
                color_rgb = green_colors[green_color]
                st.color_picker("Selected Color:", f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}", disabled=True)
            elif color_option == "Yellows & Earth Tones":
                yellow_color = st.selectbox(
                    "Select Yellow/Earth Tone:",
                    ["Marigold (21057 MR+)", "Morning Glow (22067 MR+)", "Lime (21163 MR+)", 
                     "Dark Citrus (21023 MR+)", "Parakeet (22017 MR+)"],
                    help="Warm yellow and earth tones"
                )
                yellow_colors = {
                    "Marigold (21057 MR+)": (255, 220, 100),
                    "Morning Glow (22067 MR+)": (255, 140, 80),
                    "Lime (21163 MR+)": (180, 220, 100),
                    "Dark Citrus (21023 MR+)": (120, 100, 60),
                    "Parakeet (22017 MR+)": (220, 180, 80)
                }
                color_rgb = yellow_colors[yellow_color]
                st.color_picker("Selected Color:", f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}", disabled=True)
            elif color_option == "Pinks & Reds":
                pink_color = st.selectbox(
                    "Select Pink/Red:",
                    ["Pink Salt (22104 MR+)", "Flamingo Pink (22134 MR+)", "Bubblegum (21114 MR+)", 
                     "Cardinal (21065 MR+)", "Fiesta Rose (22115 MR+)"],
                    help="Vibrant pink and red tones"
                )
                pink_colors = {
                    "Pink Salt (22104 MR+)": (255, 240, 245),
                    "Flamingo Pink (22134 MR+)": (220, 160, 180),
                    "Bubblegum (21114 MR+)": (255, 100, 180),
                    "Cardinal (21065 MR+)": (180, 40, 60),
                    "Fiesta Rose (22115 MR+)": (140, 40, 80)
                }
                color_rgb = pink_colors[pink_color]
                st.color_picker("Selected Color:", f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}", disabled=True)
            elif color_option == "Purples & Browns":
                purple_color = st.selectbox(
                    "Select Purple/Brown:",
                    ["Enigma (22172 MR+)", "Black Current (21172 MR+)", "Berry Bunch (22105 MR+)", 
                     "Shangrila (21055 MR+)", "Winter Sea (21162 MR+)"],
                    help="Rich purple and brown tones"
                )
                purple_colors = {
                    "Enigma (22172 MR+)": (100, 80, 120),
                    "Black Current (21172 MR+)": (60, 40, 50),
                    "Berry Bunch (22105 MR+)": (120, 60, 100),
                    "Shangrila (21055 MR+)": (140, 100, 80),
                    "Winter Sea (21162 MR+)": (120, 130, 150)
                }
                color_rgb = purple_colors[purple_color]
                st.color_picker("Selected Color:", f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}", disabled=True)
        elif texture_type == "Pattern Texture":
            pattern_dir = os.path.join("assets", "textures", "patterns")
            pattern_files = [f for f in os.listdir(pattern_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))] if os.path.exists(pattern_dir) else []
            if pattern_files:
                st.write("Select a pattern texture:")
                cols = st.columns(min(4, len(pattern_files)))
                selected_idx = st.radio(
                    "Pattern Texture Options:",
                    options=list(range(len(pattern_files))),
                    format_func=lambda i: pattern_files[i],
                    key="pattern_texture_radio"
                )
                for i, col in enumerate(cols):
                    with col:
                        st.image(os.path.join(pattern_dir, pattern_files[i]), caption=pattern_files[i], use_column_width=True)
                texture_path = os.path.join(pattern_dir, pattern_files[selected_idx])
                texture = PILImage.open(texture_path).convert("RGB")
                texture_name = pattern_files[selected_idx]
            else:
                st.warning("No pattern textures found. Add PNG/JPG files to assets/textures/patterns.")
                texture = PILImage.new('RGB', (512, 512), (255, 255, 255))
                texture_name = None
        elif texture_type == "Wood Texture":
            wood_dir = os.path.join("assets", "textures", "woods")
            wood_files = [f for f in os.listdir(wood_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))] if os.path.exists(wood_dir) else []
            if wood_files:
                st.write("Select a wood texture:")
                cols = st.columns(min(4, len(wood_files)))
                selected_idx = st.radio(
                    "Wood Texture Options:",
                    options=list(range(len(wood_files))),
                    format_func=lambda i: wood_files[i],
                    key="wood_texture_radio"
                )
                for i, col in enumerate(cols):
                    with col:
                        st.image(os.path.join(wood_dir, wood_files[i]), caption=wood_files[i], use_column_width=True)
                texture_path = os.path.join(wood_dir, wood_files[selected_idx])
                texture = PILImage.open(texture_path).convert("RGB")
                texture_name = wood_files[selected_idx]
            else:
                st.warning("No wood textures found. Add PNG/JPG files to assets/textures/woods.")
                texture = PILImage.new('RGB', (512, 512), (255, 255, 255))
                texture_name = None
        elif texture_type == "Upload Texture/Color":
            st.write("Upload a texture image (PNG/JPG) or pick a color:")
            uploaded_file = st.file_uploader("Upload Texture Image", type=["png", "jpg", "jpeg"], key="user_texture_upload")
            if uploaded_file is not None:
                texture = PILImage.open(uploaded_file).convert("RGB")
                texture_name = uploaded_file.name
                upload_texture_uploaded = True
                st.image(texture, caption="Uploaded Texture Preview", use_column_width=True)
            else:
                selected_color = st.color_picker("Pick a color:", "#FFFFFF")
                upload_color_rgb = tuple(int(selected_color[i:i+2], 16) for i in (1, 3, 5))
                texture = None
                texture_name = None

        # Blending Options
        st.subheader("🔀 Blending Options")
        blend_mode = st.selectbox(
            "Blend Mode:",
            ["normal", "multiply", "overlay", "photorealistic_lab"],
            help="Choose how the texture blends with the original image"
        )

        opacity = st.slider(
            "Opacity:",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="Control texture transparency"
        )

        # Debug Mode
        debug_mode = st.checkbox("🔍 Debug Mode", help="Show additional information for troubleshooting")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an interior image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of your interior space"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Full Image Preview", use_column_width=True)

            # Step 1: Show 'Select Area' button
            if 'step' not in st.session_state:
                st.session_state.step = 1
            if st.button("Select Area to Texture") or st.session_state.get('step', 1) > 1:
                st.session_state.step = 2
                st.subheader("🎯 Select Area to Texture")
                st.info("Click on the area you want to apply texture to (e.g., the wall behind the TV)")
                
                # Canvas sizing - maintain aspect ratio
                max_canvas_height = 400
                max_canvas_width = 600
                
                aspect_ratio = image.width / image.height
                
                # Calculate optimal canvas dimensions
                if aspect_ratio > (max_canvas_width / max_canvas_height):
                    # Image is wider
                    canvas_width = max_canvas_width
                    canvas_height = int(canvas_width / aspect_ratio)
                else:
                    # Image is taller or square
                    canvas_height = max_canvas_height
                    canvas_width = int(canvas_height * aspect_ratio)
                
                # Ensure minimum dimensions
                canvas_width = max(200, min(canvas_width, max_canvas_width))
                canvas_height = max(150, min(canvas_height, max_canvas_height))
                
                # Create canvas image with proper format
                try:
                    # Ensure the image is in RGB format
                    rgb_image = image.convert("RGB")
                    # Resize with high quality
                    canvas_image = rgb_image.resize((canvas_width, canvas_height), Image.LANCZOS)
                    
                    # Debug info
                    if debug_mode:
                        st.write(f"Original image: {image.size}, mode: {image.mode}")
                        st.write(f"Canvas dimensions: {canvas_width}x{canvas_height}")
                        st.write(f"Canvas image: {canvas_image.size}, mode: {canvas_image.mode}")
                        
                        # Show a small preview to verify the image is correct
                        col_debug1, col_debug2 = st.columns(2)
                        with col_debug1:
                            st.image(canvas_image, caption="Canvas Image Preview", width=150)
                        with col_debug2:
                            st.write("✅ Image processed successfully for canvas")
                    
                except Exception as e:
                    st.error(f"Error processing image for canvas: {e}")
                    canvas_image = Image.new('RGB', (canvas_width, canvas_height), (240, 240, 240))
                
                # Create canvas with improved settings
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.4)",  # Orange fill with transparency
                    stroke_width=2,
                    stroke_color="#FF4500",  # Orange red stroke
                    background_color="#FFFFFF",
                    background_image=canvas_image,
                    update_streamlit=True,
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="point",
                    point_display_radius=4,
                    key="canvas",
                    display_toolbar=True  # Show toolbar for better UX
                )
                # Extract points from canvas and map to image coordinates
                def map_points_to_image(canvas_points, canvas_size, image_size):
                    scale_x = image_size[0] / canvas_size[0]
                    scale_y = image_size[1] / canvas_size[1]
                    return [(int(x * scale_x), int(y * scale_y)) for x, y in canvas_points]
                points = []
                if canvas_result.json_data is not None:
                    for obj in canvas_result.json_data["objects"]:
                        if obj["type"] == "circle":
                            points.append((int(obj["left"]), int(obj["top"])))
                if points:
                    st.success(f"✅ {len(points)} point(s) selected")
                    if debug_mode:
                        st.write("Selected points (canvas):", points)
                    mapped_points = map_points_to_image(points, (canvas_width, canvas_height), image.size)
                    if debug_mode:
                        st.write("Mapped points (image):", mapped_points)
                    # Step 2: Show 'Generate Masks' button
                    if st.button("Generate Masks") or st.session_state.get('step', 1) > 2:
                        st.session_state.step = 3
                        if not st.session_state.sam_integration.model_loaded:
                            st.error("❌ Please load a SAM model first!")
                        else:
                            with st.spinner("Processing with SAM..."):
                                # --- Robust image resizing for SAM ---
                                orig_size = image.size
                                max_side = max(image.size)
                                scale = 1024 / max_side
                                new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
                                image_resized = image.convert("RGB").resize(new_size, Image.LANCZOS)
                                image_np = np.array(image_resized).astype(np.uint8)
                                # Ensure shape is (H, W, 3)
                                if image_np.ndim == 2:
                                    image_np = np.stack([image_np]*3, axis=-1)
                                elif image_np.shape[2] == 4:
                                    image_np = image_np[:, :, :3]
                                # Scale points to resized image
                                mapped_points_resized = [(int(x * scale), int(y * scale)) for x, y in mapped_points]
                                masks, scores = st.session_state.sam_integration.generate_masks(image_np, mapped_points_resized)
                                if masks is not None:
                                    # Resize masks back to original image size for display
                                    mask_imgs = []
                                    for i, mask in enumerate(masks):
                                        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(orig_size, Image.NEAREST)
                                        red_overlay = Image.new("RGBA", orig_size, (255, 0, 0, 0))
                                        mask_alpha = mask_img.point(lambda p: 180 if p > 0 else 0)
                                        red_overlay.putalpha(mask_alpha)
                                        preview = Image.alpha_composite(image.convert("RGBA"), red_overlay)
                                        mask_imgs.append(preview)
                                    # Store masks, previews, and scores in session state
                                    st.session_state.masks = masks
                                    st.session_state.mask_imgs = mask_imgs
                                    st.session_state.mask_scores = scores
                                    st.session_state.orig_size = orig_size
                                    st.session_state.input_image = image
                                    st.session_state.selected_mask_idx = 0
                                    st.write("Select the best mask:")
                    # --- Always show mask selection UI if masks are in session state ---
                    if (
                        'masks' in st.session_state and
                        'mask_imgs' in st.session_state and
                        st.session_state.masks is not None and
                        st.session_state.mask_imgs is not None
                    ):
                        mask_imgs = st.session_state.mask_imgs
                        scores = st.session_state.mask_scores
                        orig_size = st.session_state.orig_size
                        image = st.session_state.input_image
                        st.write("Select the best mask:")
                        cols = st.columns(len(mask_imgs))
                        for i, col in enumerate(cols):
                            with col:
                                st.image(mask_imgs[i], caption=f"Option {i+1} (Score: {scores[i]:.2f})", use_column_width=True)
                        selected_mask_idx = st.radio(
                            label="Choose the mask to apply:",
                            options=list(range(len(mask_imgs))),
                            format_func=lambda x: f"Option {x+1} (Score: {scores[x]:.2f})",
                            index=st.session_state.get('selected_mask_idx', 0),
                            key="mask_radio_group"
                        )
                        st.session_state.selected_mask_idx = selected_mask_idx
                        if st.button("Apply Texture"):
                            selected_mask = st.session_state.masks[selected_mask_idx]
                            selected_mask_up = np.array(Image.fromarray((selected_mask * 255).astype(np.uint8)).resize(orig_size, Image.NEAREST)) > 127
                            st.session_state.selected_mask = selected_mask_up.astype(np.uint8)
                            st.session_state.selected_image = image
                            st.session_state.selected_texture_type = texture_type
                            st.session_state.selected_color_rgb = color_rgb if texture_type == "Solid Color" else (255,255,255)
                            st.session_state.selected_blend_mode = blend_mode
                            st.session_state.selected_opacity = opacity
                            st.session_state.texture = texture if (texture_type != "Solid Color" and (texture is not None or upload_texture_uploaded)) else None
                            st.session_state.texture_name = texture_name
                            if texture_type == "Upload Texture/Color" and not upload_texture_uploaded:
                                st.session_state.upload_color_rgb = upload_color_rgb
                            st.session_state.step = 4
                            st.success("Mask selected! Texture will be applied in the Results panel.")
                else:
                    st.warning("⚠️ Please click on the image to select points")

    with col2:
        st.header("📊 Results")
        # Display results if available
        if (
            st.session_state.get('step', 1) == 4 and
            'selected_mask' in st.session_state and
            'selected_image' in st.session_state and
            st.session_state.selected_mask is not None and
            st.session_state.selected_image is not None
        ):
            st.subheader("✨ Textured Result")
            # Create texture
            if st.session_state.selected_texture_type == "Solid Color":
                texture = TextureApplicator.create_solid_color_texture(
                    st.session_state.selected_color_rgb, st.session_state.selected_image.size
                )
                is_tiled = False
            elif st.session_state.selected_texture_type == "Upload Texture/Color":
                if 'texture' in st.session_state and st.session_state.texture is not None:
                    texture = st.session_state.texture
                    is_tiled = True
                else:
                    # Use the uploaded color if no image was uploaded
                    texture = TextureApplicator.create_solid_color_texture(
                        st.session_state.get('upload_color_rgb', (255, 255, 255)),
                        st.session_state.selected_image.size
                    )
                    is_tiled = False
            else:
                if 'texture' in st.session_state and st.session_state.texture is not None:
                    texture = st.session_state.texture
                    is_tiled = True
                else:
                    texture = TextureApplicator.create_solid_color_texture((255, 255, 255), st.session_state.selected_image.size)
                    is_tiled = False
            # Apply texture
            result_image = TextureApplicator.apply_texture_to_mask(
                st.session_state.selected_image,
                st.session_state.selected_mask,
                texture,
                st.session_state.selected_blend_mode,
                st.session_state.selected_opacity,
                is_tiled
            )
            st.session_state.result_image = result_image
            st.image(result_image, caption="Result with Applied Texture", use_column_width=True)
            # Download button
            download_link = get_image_download_link(
                result_image, 
                "atulayakriti_result.png"
            )
            st.markdown(download_link, unsafe_allow_html=True)
            # Show mask if debug mode is on
            if debug_mode:
                st.subheader("🎭 Selected Mask")
                mask_image = Image.fromarray((st.session_state.selected_mask * 255).astype(np.uint8))
                st.image(mask_image, caption="Segmentation Mask", use_column_width=True)
        else:
            st.info("Upload an image, select area, generate masks, and apply texture to see results here")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🎨 <strong>AtulayaAkriti Texture Rendering Tool</strong> - Professional Interior Design Visualization</p>
        <p>Powered by Meta's Segment Anything Model (SAM) and Advanced Computer Vision</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
