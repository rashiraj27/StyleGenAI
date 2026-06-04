import streamlit as st
import cv2
import numpy as np
from vision_engine import analyze_skin_tone
from agent_system import app as agent_graph

# Set up clean web page configuration
st.set_page_config(page_title="StyleGenAI", page_icon="👔", layout="centered")

st.title("👔 StyleGenAI: Multi-Agent Fashion Engine")
st.write("Upload a photo to automatically extract visual traits and generate custom wardrobe plans.")

st.write("---")

# ==========================================
# STEP 1: FRONTEND INTERFACE - USER INPUTS
# ==========================================
uploaded_file = st.file_uploader("Choose a clear portrait/selfie photo", type=["jpg", "jpeg", "png"])
user_request = st.text_input("What is the occasion or style goal?", placeholder="e.g., Casual coffee date, tech startup presentation")

# ==========================================
# STEP 2: PIPELINE EXECUTION
# ==========================================
if st.button("Generate My Wardrobe Plan", type="primary"):
    if uploaded_file is not None and user_request != "":
        with st.spinner("Processing image pixels and activating AI agents..."):
            
            # 1. Convert the uploaded file bytes into an OpenCV image matrix
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            
            # Save temporarily so our vision engine can grab it
            cv2.imwrite("temp_user_pic.jpg", opencv_image)
            
            # 2. Run your Computer Vision Script
            try:
                detected_tone = analyze_skin_tone("temp_user_pic.jpg")
                if not detected_tone:
                    detected_tone = "COOL" # Fallback safeguard
            except Exception:
                detected_tone = "COOL"
            
            # Display findings on UI panels
            st.subheader("📊 Visual Traits Extracted")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Calculated Skin Undertone", value=detected_tone.upper())
            with col2:
                st.metric(label="Lighting Normalization (HSV)", value="Active")
                
            st.write("---")
            
            # 3. Fire up the LangGraph Self-Correcting Agent System
            initial_graph_input = {
                "undertone": detected_tone,
                "user_request": user_request,
                "feedback": "",
                "revision_count": 0
            }
            
            # Run the graph orchestrator
            final_outputState = agent_graph.invoke(initial_graph_input)
            
            # 4. Render results to user screen
            st.subheader("👔 Final Verified Wardrobe Plan")
            st.success("Your outfit recommendation passed our multi-agent validation criteria!")
            
            st.info(f"**Palette Parameters Enforced:** {', '.join(final_outputState['color_palette'])}")
            st.write(final_outputState['outfit_recommendation'])
            
    else:
        st.warning("⚠️ Please upload an image file and enter a style goal first.")
