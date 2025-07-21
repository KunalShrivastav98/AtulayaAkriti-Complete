# Simple coordinate selector for texture application
import streamlit as st
from PIL import Image
import numpy as np

def simple_area_selector(image):
    """Simple area selection without canvas dependency - improved for better UX"""
    st.subheader("🎯 Select Area to Texture (Alternative Method)")
    
    # Display image for reference
    st.image(image, caption=f"Image Size: {image.size[0]}x{image.size[1]} pixels", use_column_width=True)
    
    st.info("💡 Since visual canvas selection isn't available, use one of these methods to select the area you want to texture:")
    
    # Selection method
    method = st.radio(
        "Choose how to select the area:",
        ["Quick Presets", "Manual Coordinates", "Grid Selection"],
        help="Different ways to specify the area you want to apply texture to"
    )
    
    w, h = image.size
    selected_points = []
    
    if method == "Quick Presets":
        preset = st.selectbox(
            "Select a common area:",
            [
                "Center of image",
                "Left wall area", 
                "Right wall area",
                "Top area (ceiling/upper wall)",
                "Bottom area (floor)",
                "Background wall (center-back)"
            ]
        )
        
        preset_coords = {
            "Center of image": [(w//2, h//2)],
            "Left wall area": [(w//4, h//2)],
            "Right wall area": [(3*w//4, h//2)],
            "Top area (ceiling/upper wall)": [(w//2, h//4)],
            "Bottom area (floor)": [(w//2, 3*h//4)],
            "Background wall (center-back)": [(w//2, h//3)]
        }
        
        if st.button("✅ Use This Preset", type="primary"):
            selected_points = preset_coords[preset]
            st.success(f"Selected: {preset}")
            st.info(f"Coordinates: {selected_points}")
            return selected_points
    
    elif method == "Manual Coordinates":
        st.info("Enter the X,Y coordinates of the area you want to texture")
        
        col1, col2 = st.columns(2)
        with col1:
            x = st.number_input("X coordinate", min_value=0, max_value=w-1, value=w//2)
        with col2:
            y = st.number_input("Y coordinate", min_value=0, max_value=h-1, value=h//2)
        
        if st.button("✅ Set Point", type="primary"):
            selected_points = [(int(x), int(y))]
            st.success(f"Point selected at: ({x}, {y})")
            return selected_points
    
    else:  # Grid Selection
        st.info("Click on a grid section to select that area")
        
        # Create a 3x3 grid
        grid_size = 3
        cell_w = w // grid_size
        cell_h = h // grid_size
        
        # Display grid options
        for row in range(grid_size):
            cols = st.columns(grid_size)
            for col in range(grid_size):
                with cols[col]:
                    cell_name = f"Row {row+1}, Col {col+1}"
                    center_x = col * cell_w + cell_w // 2
                    center_y = row * cell_h + cell_h // 2
                    
                    if st.button(f"Select\n{cell_name}", key=f"grid_{row}_{col}"):
                        selected_points = [(center_x, center_y)]
                        st.success(f"Selected {cell_name}")
                        st.info(f"Coordinates: ({center_x}, {center_y})")
                        return selected_points
    
    return None

def generate_masks_optimized(sam_integration, image, points, debug_mode=False):
    """Generate masks with memory optimization for Streamlit Cloud"""
    
    if not sam_integration.model_loaded:
        st.error("❌ SAM model not loaded!")
        return None, None
    
    try:
        with st.spinner("🧠 Processing with SAM (optimized for cloud)..."):
            # Aggressive size reduction for Streamlit Cloud
            orig_size = image.size
            max_size = 512  # Much smaller than before
            
            max_side = max(image.size)
            if max_side > max_size:
                scale = max_size / max_side
                new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            else:
                scale = 1.0
                new_size = image.size
            
            st.info(f"Processing at {new_size[0]}x{new_size[1]} pixels (reduced from {orig_size[0]}x{orig_size[1]})")
            
            # Process image
            image_resized = image.convert("RGB").resize(new_size, Image.LANCZOS)
            image_np = np.array(image_resized).astype(np.uint8)
            
            # Ensure correct format
            if image_np.ndim == 2:
                image_np = np.stack([image_np]*3, axis=-1)
            elif image_np.shape[2] == 4:
                image_np = image_np[:, :, :3]
            
            # Scale points
            scaled_points = [(int(x * scale), int(y * scale)) for x, y in points]
            
            if debug_mode:
                st.write(f"Scaled points: {scaled_points}")
                st.write(f"Image shape: {image_np.shape}")
            
            # Generate masks
            masks, scores = sam_integration.generate_masks(image_np, scaled_points)
            
            if masks is not None and len(masks) > 0:
                # Create preview images
                mask_imgs = []
                for i, mask in enumerate(masks):
                    # Resize mask back to original size
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(orig_size, Image.NEAREST)
                    
                    # Create overlay
                    red_overlay = Image.new("RGBA", orig_size, (255, 0, 0, 0))
                    mask_alpha = mask_img.point(lambda p: 100 if p > 0 else 0)  # Lower opacity
                    red_overlay.putalpha(mask_alpha)
                    
                    # Combine with original
                    preview = Image.alpha_composite(image.convert("RGBA"), red_overlay)
                    mask_imgs.append(preview)
                
                return masks, mask_imgs, scores
            else:
                st.error("❌ No masks generated. Try different coordinates.")
                return None, None, None
                
    except Exception as e:
        st.error(f"❌ Mask generation failed: {e}")
        st.info("💡 Try using simpler coordinates or refresh the page")
        if debug_mode:
            st.exception(e)
        return None, None, None
