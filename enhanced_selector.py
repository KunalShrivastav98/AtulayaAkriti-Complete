# Enhanced image selector for HuggingFace Spaces (16GB RAM)
import streamlit as st
from PIL import Image
import numpy as np
import io
import base64
import gc

def create_clickable_image_selector(image):
    """Create a clickable image that works reliably on any platform"""
    
    # Resize image for display (keep reasonable size)
    display_width = 700  # Slightly larger for better UX
    aspect_ratio = image.height / image.width
    display_height = int(display_width * aspect_ratio)
    
    if display_height > 500:
        display_height = 500
        display_width = int(display_height / aspect_ratio)
    
    display_image = image.resize((display_width, display_height), Image.LANCZOS)
    
    # Convert to base64 for HTML embedding
    buffered = io.BytesIO()
    display_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Enhanced HTML with better styling
    html_code = f"""
    <div style="position: relative; display: inline-block; margin: 10px;">
        <img id="clickable-image" 
             src="data:image/png;base64,{img_base64}" 
             style="cursor: crosshair; border: 3px solid #FF4500; border-radius: 12px; 
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2); transition: transform 0.2s;"
             onmouseover="this.style.transform='scale(1.02)'"
             onmouseout="this.style.transform='scale(1)'"
             onclick="captureClick(event)"
        />
        <div id="click-indicator" style="position: absolute; width: 12px; height: 12px; 
             background: #FF4500; border: 2px solid white; border-radius: 50%; 
             display: none; margin: -6px; box-shadow: 0 2px 4px rgba(0,0,0,0.5);"></div>
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
        
        // Show enhanced click indicator
        const indicator = document.getElementById('click-indicator');
        indicator.style.left = x + 'px';
        indicator.style.top = y + 'px';
        indicator.style.display = 'block';
        
        // Add pulse animation
        indicator.style.animation = 'pulse 0.5s ease-in-out';
        
        // Store coordinates
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
    
    <style>
    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.5); }}
        100% {{ transform: scale(1); }}
    }}
    </style>
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

def enhanced_area_selector(image):
    """Enhanced area selector for HuggingFace Spaces"""
    st.subheader("🎯 Select Area to Texture")
    
    st.info("👆 **Click directly on the area** in the image below that you want to texture")
    
    # Try clickable image first
    selected_points = create_clickable_image_selector(image)
    
    if selected_points:
        st.success(f"✅ Point selected at: {selected_points[0]}")
        st.info("🚀 **Enhanced Processing**: Using full 1024px resolution for maximum quality!")
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

def full_quality_sam_processing(sam_integration, image, points, debug_mode=False):
    """Full quality SAM processing for HuggingFace Spaces (16GB RAM)"""
    
    if not sam_integration.model_loaded:
        st.error("❌ SAM model not loaded!")
        return None, None, None
    
    try:
        with st.spinner("🧠 Processing with SAM (Full Quality - 1024px)..."):
            # RESTORED: Full resolution processing
            orig_size = image.size
            max_size = 1024  # 🚀 ORIGINAL QUALITY RESTORED!
            
            max_side = max(image.size)
            if max_side > max_size:
                scale = max_size / max_side
                new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            else:
                scale = 1.0
                new_size = image.size
            
            st.success(f"🎨 **Full Quality Processing**: {new_size[0]}x{new_size[1]} pixels (Original: {orig_size[0]}x{orig_size[1]})")
            
            # Process image with original quality
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
            
            # Generate masks with full processing power
            masks, scores = sam_integration.generate_masks(image_np, scaled_points)
            
            if masks is not None and len(masks) > 0:
                # Limit to reasonable number but allow more than before
                max_masks = min(len(masks), 10)  # Increased from 3 to 10
                masks = masks[:max_masks]
                scores = scores[:max_masks] if scores is not None else None
                
                # Create high-quality preview images
                mask_imgs = []
                for i, mask in enumerate(masks):
                    # Resize mask back to original size with high quality
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(
                        orig_size, Image.LANCZOS  # Better resampling
                    )
                    
                    # Create enhanced overlay
                    red_overlay = Image.new("RGBA", orig_size, (255, 0, 0, 0))
                    mask_alpha = mask_img.point(lambda p: 120 if p > 0 else 0)  # Better visibility
                    red_overlay.putalpha(mask_alpha)
                    
                    # Combine with original
                    preview = Image.alpha_composite(image.convert("RGBA"), red_overlay)
                    mask_imgs.append(preview)
                
                # Efficient memory management (but less aggressive)
                if len(masks) > 5:
                    gc.collect()
                
                st.success(f"🎉 **Generated {len(masks)} high-quality masks!**")
                return masks, mask_imgs, scores
            else:
                st.error("❌ No masks generated. Try different coordinates.")
                return None, None, None
                
    except Exception as e:
        st.error(f"❌ Mask generation failed: {e}")
        st.info("💡 Try using different coordinates or refresh the page")
        if debug_mode:
            st.exception(e)
        return None, None, None
