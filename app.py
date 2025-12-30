import streamlit as st
from PIL import Image
import json
import torch
import os
import sys

# Add src to python path to import inference module
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from inference import predict, load_model

# Constants
MODEL_PATH = "src/models/best_model.pth"
CLASS_NAMES_PATH = "configs/flower_class_names.json"

st.set_page_config(page_title="Flower Classifier", page_icon="🌸")

@st.cache_resource
def get_cached_model():
    """
    Loads and caches the model to avoid reloading on every interaction.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_path check handled in load_model but let's be safe
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        return None, None
        
    try:
        model = load_model(MODEL_PATH, device=device)
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_data
def load_class_names():
    if not os.path.exists(CLASS_NAMES_PATH):
        st.error(f"Class names file not found: {CLASS_NAMES_PATH}")
        return {}
    with open(CLASS_NAMES_PATH, 'r') as f:
        # The file is a JSON list of names, index corresponds to class ID
        names = json.load(f)
    return names

def main():
    st.title("🌸 Flower Classification App")
    st.write("Upload an image of a flower to identify its species.")

    # Load resources
    model, device = get_cached_model()
    class_names = load_class_names()

    if model is None or not class_names:
        st.warning("Please ensure model and class names files are present in the directory.")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            
            # Layout: Image on left, results on right
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption='Uploaded Image', use_container_width=True)
                
            with col2:
                with st.spinner('Classifying...'):
                    probs, indices = predict(image, model, device=device)
                
                st.subheader("Results:")
                
                # Display top 3 predictions
                for i in range(len(probs)):
                    class_id = indices[i]
                    # Handle json format: if list, index directly. If dict, use key. 
                    # Based on grep output, it looks like a list: ["pink primrose", ...]
                    if isinstance(class_names, list):
                        if class_id < len(class_names):
                            class_name = class_names[class_id]
                        else:
                            class_name = f"Class {class_id}"
                    else:
                        class_name = class_names.get(str(class_id), f"Class {class_id}")
                        
                    prob = probs[i]
                    
                    st.write(f"**{class_name}** ({prob*100:.1f}%)")
                    st.progress(float(prob))
                    
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
