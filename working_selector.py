# Working image clickable selector for Streamlit Cloud
import streamlit as st
from PIL import Image
import numpy as np
import io
import base64

def create_clickable_image_selector(image):
    """Create a clickable image that works reliably on Streamlit Cloud"""
    
    # Resize image for display (keep reasonable size)
    display_width = 600
    aspect_ratio = image.height / image.width
    display_height = int(display_width * aspect_ratio)
    
    if display_height > 400:
        display_height = 400
        display_width = int(display_height / aspect_ratio)
    
    display_image = image.resize((display_width, display_height), Image.LANCZOS)
    
    # Convert to base64 for HTML embedding
    buffered = io.BytesIO()
    display_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Create HTML with click detection
    html_code = f"""
    <div style="position: relative; display: inline-block;">
        <img id="clickable-image" 
             src="data:image/png;base64,{img_base64}" 
             style="cursor: crosshair; border: 2px solid #FF4500; border-radius: 8px;"
             onclick="captureClick(event)"
        />
        <div id="click-indicator" style="position: absolute; width: 10px; height: 10px; 
             background: #FF4500; border-radius: 50%; display: none; margin: -5px;"></div>
    </div>
    
    <script>
    function captureClick(event) {{
        const img = event.target;
        const rect = img.getBoundingClientRect();
        const x = Math.round(event.clientX - rect.left);
        const y = Math.round(event.clientY - rect.top);
        
        // Scale coordinates back to original image size
        const scaleX = {image.width} / {display_width};
        const scaleY = {image.height} / {display_height};
        const originalX = Math.round(x * scaleX);
        const originalY = Math.round(y * scaleY);
        
        // Show click indicator
        const indicator = document.getElementById('click-indicator');
        indicator.style.left = x + 'px';
        indicator.style.top = y + 'px';
        indicator.style.display = 'block';
        
        // Store coordinates in Streamlit session state via query params
        const url = new URL(window.location);
        url.searchParams.set('click_x', originalX);
        url.searchParams.set('click_y', originalY);
        url.searchParams.set('timestamp', Date.now());
        window.history.replaceState({{}}, '', url);
        
        // Trigger Streamlit rerun
        window.parent.postMessage({{
            type: 'streamlit:setComponentValue',
            data: {{x: originalX, y: originalY, timestamp: Date.now()}}
        }}, '*');
    }}
    </script>
    """
    
    # Display the clickable image
    st.markdown(html_code, unsafe_allow_html=True)
    
    # Check for click coordinates
    query_params = st.query_params
    if "click_x" in query_params and "click_y" in query_params:
        try:
            x = int(query_params["click_x"])
            y = int(query_params["click_y"])
            
            # Clear the query params
            st.query_params.clear()
            
            return [(x, y)]
        except:
            pass
    
    return None

def working_area_selector(image):
    """Working area selector that bypasses canvas issues"""
    st.subheader("🎯 Select Area to Texture")
    
    st.info("👆 **Click directly on the area** in the image below that you want to texture")
    
    # Try clickable image first
    selected_points = create_clickable_image_selector(image)
    
    if selected_points:
        st.success(f"✅ Point selected at: {selected_points[0]}")
        return selected_points
    
    # Fallback to coordinate input
    st.markdown("---")
    st.write("**Alternative: Enter coordinates manually**")
    
    col1, col2 = st.columns(2)
    with col1:
        x = st.number_input("X coordinate", min_value=0, max_value=image.width-1, value=image.width//2)
    with col2:
        y = st.number_input("Y coordinate", min_value=0, max_value=image.height-1, value=image.height//2)
    
    if st.button("📍 Use These Coordinates", type="primary"):
        return [(int(x), int(y))]
    
    return None

def ultra_lightweight_sam_processing(sam_integration, image, points, debug_mode=False):
    """Ultra memory-optimized SAM processing for Streamlit Cloud"""
    
    if not sam_integration.model_loaded:
        st.error("❌ SAM model not loaded!")
        return None, None, None
    
    try:
        with st.spinner("🧠 Processing with SAM (ultra-optimized)..."):
            # Extreme size reduction for Streamlit Cloud
            orig_size = image.size
            max_size = 384  # Even smaller than before
            
            max_side = max(image.size)
            if max_side > max_size:
                scale = max_size / max_side
                new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            else:
                scale = 1.0
                new_size = image.size
            
            st.info(f"Ultra-optimized processing: {new_size[0]}x{new_size[1]} pixels")
            
            # Process image with memory cleanup
            image_resized = image.convert("RGB").resize(new_size, Image.LANCZOS)
            image_np = np.array(image_resized).astype(np.uint8)
            
            # Ensure correct format and clean up
            if image_np.ndim == 2:
                image_np = np.stack([image_np]*3, axis=-1)
            elif image_np.shape[2] == 4:
                image_np = image_np[:, :, :3]
            
            # Scale points
            scaled_points = [(int(x * scale), int(y * scale)) for x, y in points]
            
            if debug_mode:
                st.write(f"Scaled points: {scaled_points}")
            
            # Generate masks with memory management
            try:
                masks, scores = sam_integration.generate_masks(image_np, scaled_points)
                
                # Force garbage collection
                import gc
                gc.collect()
                
                if masks is not None and len(masks) > 0:
                    # Create previews with minimal memory usage
                    mask_imgs = []
                    for i, mask in enumerate(masks[:3]):  # Limit to 3 masks max
                        # Create preview with lower opacity for memory efficiency
                        mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(orig_size, Image.NEAREST)
                        
                        # Simple overlay without RGBA conversion
                        preview_array = np.array(image.convert("RGB"))
                        mask_array = np.array(mask_resized) > 127
                        
                        # Apply red tint to masked area
                        preview_array[mask_array] = (
                            preview_array[mask_array] * 0.7 + 
                            np.array([255, 0, 0]) * 0.3
                        ).astype(np.uint8)
                        
                        preview = Image.fromarray(preview_array)
                        mask_imgs.append(preview)
                        
                        # Clean up intermediate arrays
                        del mask_resized, preview_array, mask_array
                        gc.collect()
                    
                    return masks[:3], mask_imgs, scores[:3]
                else:
                    st.error("❌ No masks generated. Try a different point.")
                    return None, None, None
                    
            except Exception as e:
                st.error(f"❌ SAM processing failed: {e}")
                st.info("💡 Try refreshing the page or using a smaller image")
                return None, None, None
                
    except Exception as e:
        st.error(f"❌ Mask generation failed: {e}")
        st.info("💡 The image might be too large. Try uploading a smaller image.")
        if debug_mode:
            st.exception(e)
        return None, None, None
