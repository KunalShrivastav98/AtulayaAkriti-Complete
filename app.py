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
</style>
""", unsafe_allow_html=True)

class SAMIntegration:
    """Segment Anything Model integration for HuggingFace Spaces"""
    
    def __init__(self):
        self.model_loaded = False
        self.sam_predictor = None
        self.device = "cuda" if torch.cuda.is_available() and TORCH_AVAILABLE else "cpu"
        
    def download_sam_model(self):
        """Download SAM model if not present"""
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        model_path = os.path.join(model_dir, "sam_vit_b_01ec64.pth")
        
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            if file_size > 300 * 1024 * 1024:  # 300MB minimum
                st.success(f"✅ SAM model found! Size: {file_size / (1024*1024):.1f}MB")
                return model_path
            else:
                logger.warning(f"Model file incomplete: {file_size / (1024*1024):.1f}MB")
        
        # Download model
        model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        
        try:
            st.info("📥 Downloading SAM model (358MB)... This may take a few minutes.")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(model_path, 'wb') as f:
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
            st.success("🎉 SAM model downloaded successfully!")
            return model_path
            
        except Exception as e:
            st.error(f"❌ Error downloading SAM model: {e}")
            logger.error(f"Model download error: {e}")
            return None

    def load_sam_model(self):
        """Load the SAM model"""
        if not TORCH_AVAILABLE:
            st.error("❌ PyTorch not available. Please check your environment.")
            return False
            
        try:
            # Download model if needed
            model_path = self.download_sam_model()
            if not model_path:
                return False
            
            st.info("🔄 Loading SAM model...")
            
            # Import SAM here to avoid issues if not available
            try:
                from segment_anything import sam_model_registry, SamPredictor
            except ImportError:
                st.error("❌ Segment Anything not installed. Please install: pip install segment-anything")
                return False
            
            # Load model
            model_type = "vit_b"
            sam = sam_model_registry[model_type](checkpoint=model_path)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
            self.model_loaded = True
            
            st.success(f"✅ SAM model loaded successfully! ({model_type} on {self.device})")
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

def main():
    """Main application function"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎨 AtulayaAkriti - Enhanced Texture Rendering</h1>
        <p>Professional texture application with AI-powered segmentation • Now on HuggingFace Spaces with 16GB RAM!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize SAM
    if 'sam_integration' not in st.session_state:
        st.session_state.sam_integration = SAMIntegration()
    
    sam_integration = st.session_state.sam_integration
    
    # Load model if not loaded
    if not sam_integration.model_loaded:
        if st.button("🧠 Load SAM Model", type="primary"):
            sam_integration.load_sam_model()
    
    if not sam_integration.model_loaded:
        st.warning("🔄 Please load the SAM model first to start using the texture application tool.")
        st.info("💡 The model will be downloaded automatically (358MB) - this is a one-time process.")
        return
    
    # File upload
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a PNG, JPG, or JPEG image to apply textures to"
    )
    
    if uploaded_file is not None:
        # Load image
        uploaded_image = Image.open(uploaded_file).convert("RGB")
        
        # Display image info
        st.success(f"✅ Image loaded: {uploaded_image.size[0]}x{uploaded_image.size[1]} pixels")
        
        # Show original image
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("🖼️ Original Image")
            st.image(uploaded_image, caption="Original Image", use_column_width=True)
        
        with col2:
            st.subheader("ℹ️ Image Info")
            st.write(f"**Size:** {uploaded_image.size[0]} x {uploaded_image.size[1]} pixels")
            st.write(f"**Format:** {uploaded_image.format if hasattr(uploaded_image, 'format') else 'Unknown'}")
            st.write(f"**Mode:** {uploaded_image.mode}")
        
        # Area Selection
        if SELECTOR_AVAILABLE:
            selected_points = working_area_selector(uploaded_image)
        else:
            st.error("❌ Selector module not available. Please check file uploads.")
            selected_points = None
        
        # Process masks if points are selected
        if selected_points:
            if st.button("🧠 Generate Mask", type="primary"):
                with st.spinner("Processing with SAM..."):
                    if SELECTOR_AVAILABLE:
                        masks, mask_imgs, scores = ultra_lightweight_sam_processing(
                            sam_integration, uploaded_image, selected_points, debug_mode=False
                        )
                    else:
                        st.error("❌ Processing functions not available.")
                        masks, mask_imgs, scores = None, None, None
                
                if masks is not None and len(masks) > 0:
                    st.success(f"🎉 Generated {len(masks)} masks!")
                    
                    # Store in session state
                    st.session_state.masks = masks
                    st.session_state.mask_imgs = mask_imgs
                    st.session_state.scores = scores
                    st.session_state.uploaded_image = uploaded_image
        
        # Display masks and texture application
        if 'masks' in st.session_state and st.session_state.masks is not None:
            st.subheader("🎭 Generated Masks")
            
            # Mask selection
            cols = st.columns(min(len(st.session_state.mask_imgs), 4))
            selected_mask_idx = None
            
            for i, (col, mask_img) in enumerate(zip(cols, st.session_state.mask_imgs)):
                with col:
                    st.image(mask_img, caption=f"Mask {i+1}", use_column_width=True)
                    if scores := st.session_state.scores:
                        st.write(f"Score: {scores[i]:.3f}")
                    if st.button(f"Select Mask {i+1}", key=f"mask_{i}"):
                        selected_mask_idx = i
            
            # Texture application
            if selected_mask_idx is not None:
                st.subheader("🎨 Apply Texture")
                st.success(f"Selected Mask {selected_mask_idx + 1}")
                
                # Simple texture options
                texture_type = st.selectbox(
                    "Choose texture type:",
                    ["Solid Color", "Wood Grain", "Pattern"]
                )
                
                if texture_type == "Solid Color":
                    color = st.color_picker("Pick a color", "#FF6B6B")
                    color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                    
                    if st.button("Apply Color", type="primary"):
                        # Simple color application
                        mask = st.session_state.masks[selected_mask_idx]
                        result_image = st.session_state.uploaded_image.copy()
                        
                        # Convert to numpy for processing
                        img_array = np.array(result_image)
                        
                        # Apply color where mask is True
                        img_array[mask] = color_rgb
                        
                        result_image = Image.fromarray(img_array)
                        
                        st.subheader("🎉 Result")
                        st.image(result_image, caption="Textured Image", use_column_width=True)
                        
                        # Download link
                        buffered = io.BytesIO()
                        result_image.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        href = f'<a href="data:image/png;base64,{img_str}" download="textured_image.png" class="download-button">📥 Download Result</a>'
                        st.markdown(href, unsafe_allow_html=True)
                
                else:
                    st.info("🚧 Advanced texture options (Wood Grain, Patterns) will be available after uploading texture assets.")

if __name__ == "__main__":
    main()
