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

# Configure Streamlit for HuggingFace
st.set_page_config(
    page_title="AtulayaAkriti Texture Rendering Tool",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import torch with error handling
try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PyTorch not available: {e}")
    TORCH_AVAILABLE = False

# Import selectors
try:
    from working_selector import working_area_selector, ultra_lightweight_sam_processing
    from enhanced_selector import create_enhanced_clickable_selector, full_quality_sam_processing
    SELECTOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Selector not available: {e}")
    SELECTOR_AVAILABLE = False

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E7D32 0%, #FFA726 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background: linear-gradient(90deg, #2E7D32 0%, #FFA726 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .texture-preview {
        border: 2px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        margin: 5px;
        transition: border-color 0.3s;
    }
    .texture-preview:hover {
        border-color: #FFA726;
    }
    .download-button {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        color: white;
        text-decoration: none;
        padding: 10px 20px;
        border-radius: 5px;
        display: inline-block;
        margin: 10px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    .download-button:hover {
        transform: translateY(-2px);
        text-decoration: none;
        color: white;
    }
    .warning-message {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class SAMIntegration:
    """Enhanced SAM integration for HuggingFace Spaces with 16GB RAM"""
    
    def __init__(self):
        self.model_loaded = False
        self.sam_predictor = None
        self.device = "cuda" if torch.cuda.is_available() and TORCH_AVAILABLE else "cpu"
        self.available_models = self.check_available_models()
        
    def check_available_models(self):
        """Check for available SAM models"""
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            return []
        
        model_files = []
        for file in os.listdir(model_dir):
            if file.endswith('.pth'):
                file_path = os.path.join(model_dir, file)
                file_size = os.path.getsize(file_path)
                # Check if file is complete (at least 300MB)
                if file_size > 300 * 1024 * 1024:
                    model_files.append(file)
        
        return model_files
        
    def auto_download_model(self):
        """Auto-download SAM ViT-B model for HuggingFace"""
        model_filename = os.path.join("models", "sam_vit_b_01ec64.pth")
        model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        
        try:
            # Check if model already exists and is complete
            if os.path.exists(model_filename):
                file_size = os.path.getsize(model_filename)
                if file_size > 350 * 1024 * 1024:  # At least 350MB
                    st.success(f"✅ SAM model already available! Size: {file_size / (1024*1024):.1f}MB")
                    self.available_models = self.check_available_models()
                    return True
                else:
                    # Remove incomplete file
                    try:
                        os.remove(model_filename)
                        st.info("🗑️ Removed corrupted model file")
                    except:
                        pass
            
            st.info("🔄 Downloading SAM ViT-B model (358MB) - optimized for HuggingFace Spaces...")
            
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
            st.success("✅ SAM ViT-B model downloaded successfully for HuggingFace!")
            
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
            except ImportError:
                st.error("❌ Segment Anything not installed. Install with: pip install segment-anything")
                return False
            
            # Use default model if none specified
            if model_path is None:
                if not self.available_models:
                    st.error("❌ No SAM models available. Please download a model first.")
                    return False
                model_path = f"models/{self.available_models[0]}"
            
            if not os.path.exists(model_path):
                st.error(f"❌ Model file not found: {model_path}")
                return False
            
            # Determine model type from filename
            if "vit_h" in model_path:
                model_type = "vit_h"
            elif "vit_l" in model_path:
                model_type = "vit_l"
            else:
                model_type = "vit_b"
            
            st.info(f"🔄 Loading SAM {model_type.upper()} model... (16GB RAM available)")
            
            # Load model
            sam = sam_model_registry[model_type](checkpoint=model_path)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
            self.model_loaded = True
            
            st.success(f"✅ SAM {model_type.upper()} model loaded successfully! (Device: {self.device})")
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading SAM model: {e}")
            logger.error(f"SAM loading error: {e}")
            return False

    def generate_masks(self, image: np.ndarray, points: List[Tuple[int, int]]):
        """Generate masks with full quality on HuggingFace Spaces"""
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
    """Advanced texture application with professional blending"""
    
    @staticmethod
    def tile_texture_to_image(texture: Image.Image, target_size: tuple) -> Image.Image:
        """Tile texture to cover target image size"""
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
                
                for c in [1, 2]:  # Only blend A and B channels, keep L
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
        <h1>🎨 AtulayaAkriti - Professional Texture Rendering</h1>
        <p>AI-Powered Interior Design • Full Quality on HuggingFace Spaces with 16GB RAM!</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize SAM
    if 'sam_integration' not in st.session_state:
        st.session_state.sam_integration = SAMIntegration()
        
        # Auto-download if no models available
        if not st.session_state.sam_integration.available_models:
            st.info("🚀 First-time setup on HuggingFace Spaces - downloading SAM model...")
            with st.spinner("Downloading SAM model (358MB) - leveraging 16GB RAM..."):
                if st.session_state.sam_integration.auto_download_model():
                    st.success("✅ Model downloaded successfully! App ready for full-quality processing.")
                    st.rerun()
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
            if st.button("🚀 Auto-Download ViT-B Model (358MB)", help="Automatically download the recommended model"):
                if st.session_state.sam_integration.auto_download_model():
                    st.rerun()
            
            st.markdown("**Or manually download:**")
            st.markdown("""
            <div class="warning-message">
                <strong>Manual Download Options:</strong><br>
                • <a href="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" target="_blank">ViT-B (358MB)</a><br>
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

        texture = None
        texture_name = None
        color_rgb = (255, 255, 255)
        
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
                selected_idx = st.radio(
                    "Pattern Texture Options:",
                    options=list(range(len(pattern_files))),
                    format_func=lambda i: pattern_files[i],
                    key="pattern_texture_radio"
                )
                cols = st.columns(min(4, len(pattern_files)))
                for i, col in enumerate(cols):
                    with col:
                        st.image(os.path.join(pattern_dir, pattern_files[i]), caption=pattern_files[i], use_column_width=True)
                
                texture_path = os.path.join(pattern_dir, pattern_files[selected_idx])
                texture = Image.open(texture_path).convert("RGB")
                texture_name = pattern_files[selected_idx]
            else:
                st.warning("No pattern textures found. Add PNG/JPG files to assets/textures/patterns.")

        elif texture_type == "Wood Texture":
            wood_dir = os.path.join("assets", "textures", "woods")
            wood_files = [f for f in os.listdir(wood_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))] if os.path.exists(wood_dir) else []
            if wood_files:
                st.write("Select a wood texture:")
                selected_idx = st.radio(
                    "Wood Texture Options:",
                    options=list(range(len(wood_files))),
                    format_func=lambda i: wood_files[i],
                    key="wood_texture_radio"
                )
                cols = st.columns(min(4, len(wood_files)))
                for i, col in enumerate(cols):
                    with col:
                        st.image(os.path.join(wood_dir, wood_files[i]), caption=wood_files[i], use_column_width=True)
                
                texture_path = os.path.join(wood_dir, wood_files[selected_idx])
                texture = Image.open(texture_path).convert("RGB")
                texture_name = wood_files[selected_idx]
            else:
                st.warning("No wood textures found. Add PNG/JPG files to assets/textures/woods.")

        elif texture_type == "Upload Texture/Color":
            st.write("Upload a texture image (PNG/JPG) or pick a color:")
            uploaded_file = st.file_uploader("Upload Texture Image", type=["png", "jpg", "jpeg"], key="user_texture_upload")
            if uploaded_file is not None:
                texture = Image.open(uploaded_file).convert("RGB")
                texture_name = uploaded_file.name
                st.image(texture, caption="Uploaded Texture Preview", use_column_width=True)
            else:
                selected_color = st.color_picker("Pick a color:", "#FFFFFF")
                color_rgb = tuple(int(selected_color[i:i+2], 16) for i in (1, 3, 5))

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

        is_tiled = st.checkbox("🔄 Tile Texture", value=True, help="Repeat texture to cover the area")

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

            # Step 1: Area Selection
            if 'step' not in st.session_state:
                st.session_state.step = 1
                
            if st.button("🎯 Select Area to Texture") or st.session_state.get('step', 1) > 1:
                st.session_state.step = 2
                st.subheader("🎯 Select Area to Texture")
                st.info("Click on the area you want to apply texture to (e.g., the wall behind the TV)")
                
                # Use enhanced selector for HuggingFace with 16GB RAM
                if SELECTOR_AVAILABLE:
                    try:
                        selected_points = create_enhanced_clickable_selector(image)
                    except:
                        # Fallback to working selector
                        selected_points = working_area_selector(image)
                else:
                    st.error("❌ Selector modules not available. Please check file uploads.")
                    selected_points = None
                
                # Process masks if points are selected
                if selected_points:
                    st.success(f"✅ {len(selected_points)} point(s) selected")
                    if debug_mode:
                        st.write("Selected points:", selected_points)
                    
                    # Step 2: Generate Masks
                    if st.button("🧠 Generate Masks (Full Quality)", type="primary") or st.session_state.get('step', 1) > 2:
                        st.session_state.step = 3
                        if not st.session_state.sam_integration.model_loaded:
                            st.error("❌ Please load a SAM model first!")
                        else:
                            # Use full quality processing on HuggingFace Spaces
                            if SELECTOR_AVAILABLE:
                                try:
                                    masks, mask_imgs, scores = full_quality_sam_processing(
                                        st.session_state.sam_integration, 
                                        image, 
                                        selected_points, 
                                        debug_mode
                                    )
                                except:
                                    # Fallback to ultra lightweight
                                    masks, mask_imgs, scores = ultra_lightweight_sam_processing(
                                        st.session_state.sam_integration, 
                                        image, 
                                        selected_points, 
                                        debug_mode
                                    )
                            else:
                                st.error("❌ Processing functions not available.")
                                masks, mask_imgs, scores = None, None, None
                            
                            if masks is not None and mask_imgs is not None:
                                # Store results in session state
                                st.session_state.masks = masks
                                st.session_state.mask_imgs = mask_imgs
                                st.session_state.mask_scores = scores
                                st.session_state.orig_size = image.size
                                st.session_state.input_image = image
                                st.session_state.selected_mask_idx = 0

    with col2:
        st.header("🎭 Mask Selection & Texture Application")
        
        # Show mask selection if masks are available
        if (
            'masks' in st.session_state and
            'mask_imgs' in st.session_state and
            st.session_state.masks is not None and
            st.session_state.mask_imgs is not None
        ):
            mask_imgs = st.session_state.mask_imgs
            scores = st.session_state.mask_scores
            
            st.subheader("🎭 Generated Masks")
            st.success(f"🎉 Successfully generated {len(mask_imgs)} masks!")
            
            # Display masks in a grid
            cols = st.columns(min(3, len(mask_imgs)))
            selected_mask_idx = None
            
            for i, col in enumerate(cols):
                with col:
                    st.image(mask_imgs[i], caption=f"Mask {i+1}", use_column_width=True)
                    if scores is not None:
                        st.write(f"Confidence: {scores[i]:.3f}")
                    if st.button(f"Select Mask {i+1}", key=f"select_mask_{i}"):
                        st.session_state.selected_mask_idx = i
                        selected_mask_idx = i

            # Get current selection
            if 'selected_mask_idx' in st.session_state:
                selected_mask_idx = st.session_state.selected_mask_idx
                
            if selected_mask_idx is not None:
                st.success(f"✅ Selected Mask {selected_mask_idx + 1}")
                
                # Apply texture button
                if st.button("🎨 Apply Texture", type="primary"):
                    with st.spinner("Applying texture with professional blending..."):
                        mask = st.session_state.masks[selected_mask_idx]
                        original_image = st.session_state.input_image
                        
                        if texture_type == "Solid Color":
                            # Create solid color texture
                            texture_img = Image.new('RGB', (256, 256), color_rgb)
                            result_image = TextureApplicator.apply_texture_to_mask(
                                original_image, mask, texture_img, blend_mode, opacity, is_tiled
                            )
                            result_name = f"solid_color_result_{color_rgb}.png"
                            
                        elif texture is not None:
                            # Apply selected texture
                            result_image = TextureApplicator.apply_texture_to_mask(
                                original_image, mask, texture, blend_mode, opacity, is_tiled
                            )
                            result_name = f"textured_result_{texture_name}.png" if texture_name else "textured_result.png"
                            
                        else:
                            st.error("❌ No texture selected or available!")
                            result_image = None
                        
                        if result_image:
                            st.subheader("🎉 Final Result")
                            st.image(result_image, caption="Textured Image", use_column_width=True)
                            
                            # Download link
                            download_link = get_image_download_link(result_image, result_name)
                            st.markdown(download_link, unsafe_allow_html=True)
                            
                            # Store result
                            st.session_state.result_image = result_image

if __name__ == "__main__":
    main()
