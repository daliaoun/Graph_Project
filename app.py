import streamlit as st
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from seam_carving import FastSeamCarver
import time

st.set_page_config(page_title="Seam Carving Image Resizer", layout="wide", page_icon="🖼️")

# Custom CSS for better appearance
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #00cc00;
    }
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🖼️ Content-Aware Image Resizing with Seam Carving")
st.markdown("""
**Fast & Optimized** - This application uses an optimized seam carving algorithm with Numba JIT compilation 
for **up to 10x faster** processing while preserving important content.
""")

# Main upload section (not in sidebar)
st.subheader("📤 Upload Your Image")
uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Initialize FastSeamCarver for better performance
    try:
        carver = FastSeamCarver(temp_path)
    except:
        from seam_carving import SeamCarver
        carver = SeamCarver(temp_path)
    
    # Display original image info
    st.info(f"**Original Size:** {carver.width} × {carver.height} pixels")
    
    # Resize controls in main area (not sidebar)
    st.subheader("⚙️ Resize Options")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        max_reduction = min(carver.width // 2, 300)  # Limit to half width or 300px
        seams_to_remove = st.slider(
            "Number of seams to remove",
            min_value=1,
            max_value=max_reduction,
            value=min(50, max_reduction),
            help="More seams = narrower image. Optimized for fast processing!"
        )
    
    with col2:
        st.metric("New Width", f"{carver.width - seams_to_remove} px")
        st.metric("Reduction", f"{(seams_to_remove / carver.width) * 100:.1f}%")
    
    new_width = carver.width - seams_to_remove
    
    # Always show seam visualization first
    st.subheader("🔍 Seam Visualization")
    st.markdown(f"""
    The **red lines** show the **{seams_to_remove} seams** that will be removed. 
    These are the lowest-energy vertical paths through the image.
    """)
    
    with st.spinner(f"Finding {seams_to_remove} optimal seams..."):
        start_viz = time.time()
        try:
            vis_carver = FastSeamCarver(temp_path)
        except:
            from seam_carving import SeamCarver
            vis_carver = SeamCarver(temp_path)
            
        seam_image = vis_carver.visualize_seam(num_seams=seams_to_remove)
        viz_time = time.time() - start_viz
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(carver.original_image, caption="Original", use_container_width=True)
        with col2:
            st.image(seam_image, caption=f"{seams_to_remove} Seams Highlighted", use_container_width=True)
        
        st.success(f"⏱️ Visualization completed in {viz_time:.2f} seconds")
    
    # Action button
    st.markdown("---")
    process_button = st.button("🚀 Resize Image", type="primary", use_container_width=True)
    
    # Main content area
    tab1, tab2 = st.tabs(["📊 Results", "ℹ️ About"])
    
    with tab1:
        if process_button:
            # Performance tracking
            start_time = time.time()
            
            with st.spinner("Processing image with optimized algorithm..."):
                # Create a fresh carver instance for resizing
                try:
                    resize_carver = FastSeamCarver(temp_path)
                except:
                    from seam_carving import SeamCarver
                    resize_carver = SeamCarver(temp_path)
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_text = st.empty()
                
                # Resize with progress updates
                num_seams = resize_carver.width - new_width
                seams_processed = 0
                
                for i in range(num_seams):
                    seam = resize_carver.find_vertical_seam()
                    resize_carver.remove_vertical_seam(seam)
                    seams_processed += 1
                    
                    # Update progress every 5 seams or at the end
                    if seams_processed % 5 == 0 or seams_processed == num_seams:
                        progress = seams_processed / num_seams
                        progress_bar.progress(progress)
                        elapsed = time.time() - start_time
                        status_text.text(f"Removed {seams_processed}/{num_seams} seams")
                        
                        # Estimate remaining time
                        if seams_processed > 0:
                            time_per_seam = elapsed / seams_processed
                            remaining_seams = num_seams - seams_processed
                            eta = remaining_seams * time_per_seam
                            time_text.text(f"⏱️ Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
                
                resized_image = resize_carver.image
                total_time = time.time() - start_time
                
                progress_bar.empty()
                status_text.empty()
                time_text.empty()
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(carver.original_image, use_container_width=True)
                    st.caption(f"Size: {carver.width} × {carver.height} pixels")
                
                with col2:
                    st.subheader("Resized Image")
                    st.image(resized_image, use_container_width=True)
                    st.caption(f"Size: {new_width} × {carver.height} pixels")
                
                # Performance metrics
                st.success(f"✅ Resize complete in **{total_time:.2f} seconds**!")
                
                col1, col2 = st.columns(2)
                col1.metric("Processing Speed", f"{seams_to_remove/total_time:.1f} seams/sec")
                col2.metric("Time per Seam", f"{total_time/seams_to_remove*1000:.1f} ms")
                
                # Download button
                img_byte_arr = io.BytesIO()
                Image.fromarray(resized_image).save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                st.download_button(
                    label="📥 Download Resized Image",
                    data=img_byte_arr,
                    file_name=f"resized_{seams_to_remove}seams.png",
                    mime="image/png",
                    use_container_width=True
                )
        else:
            st.info("👈 Adjust settings in the sidebar and click '🚀 Resize Image' to start.")
            
            # Show original image
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(carver.original_image, caption="Original Image", use_container_width=True)
    
    with tab2:
        st.subheader("How Seam Carving Works")
        
        st.markdown("""
        ### Algorithm Overview
        
        Seam Carving is a content-aware image resizing technique that removes or adds pixels 
        based on their importance in the image. It's based on finding shortest paths in a 
        Directed Acyclic Graph (DAG).
        
        ### Steps:
        
        1. **Energy Calculation** ⚡
           - Calculate energy for each pixel using gradient magnitude
           - High energy = important features (edges, textures)
           - Low energy = uniform areas (sky, walls)
           - Uses optimized Sobel filters
        
        2. **Graph Formulation (DAG)** 📊
           - Each pixel is a node
           - Edges connect each pixel to three pixels below it:
             - Down-Left: (x-1, y+1)
             - Directly Down: (x, y+1)
             - Down-Right: (x+1, y+1)
           - Edge weights = pixel energy
        
        3. **Finding the Seam (Shortest Path)** 🔍
           - Use Dynamic Programming to find minimum energy path
           - Formula: `M(x,y) = E(x,y) + min(M(x-1,y-1), M(x,y-1), M(x+1,y-1))`
           - Topologically sorted (row by row)
           - Optimized with Numba JIT compilation
        
        4. **Seam Removal** ✂️
           - Remove the pixels along the minimum energy path
           - Shift remaining pixels to fill the gap
           - Repeat until desired width is achieved
        
        ### Why DAG?
        
        Since we only move downward (from row y to row y+1), we can never form a cycle.
        This allows us to use efficient shortest path algorithms with dynamic programming.
        
        ### Advantages:
        
        - ✅ Preserves important content
        - ✅ Works better than simple cropping or scaling
        - ✅ Maintains aspect ratio of important objects
        - ✅ Efficient O(width × height) complexity
        
        
        ---
        

        """)

else:
    st.info("� Please upload an image to get started.")
    
    # Show example/instructions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚀 Getting Started
        
        1. **Upload** an image using the uploader above
        2. **Adjust** the number of seams to remove
        3. **View** the seam visualization automatically
        4. **Click** 'Resize Image' to process
        5. **Download** your resized image!
        
        ### ⚡ Features
        
        - **Numba JIT compilation** for 10x speed
        - **Automatic seam visualization**
        - **Real-time progress tracking**
        """)
    
    with col2:
        st.markdown("""
        ### 💡 Tips for Best Results
        
        - Works best with **landscape photos**
        - Ideal for images with clear subjects
        - Recommended: **10-30% width reduction**
        - Larger images take longer to process
        
        ### 📋 Supported Formats
        
        - JPEG (.jpg, .jpeg)
        - PNG (.png)
        - BMP (.bmp)
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
<strong>⚡ Fast Seam Carving</strong><br>
BY MOHAMED ALI AOUN | M1 AIDA 2025/2026
</div>
""", unsafe_allow_html=True)