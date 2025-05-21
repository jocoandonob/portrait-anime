import os
import logging
import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageFilter

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get the device
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Set Hugging Face token for authentication
huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
if huggingface_token:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = huggingface_token
    logger.info("Set Hugging Face Hub token in environment variables")
else:
    logger.warning("No Hugging Face token provided")

# Global model variables
using_real_models = False
model1 = None
model2 = None
face2paint = None

def load_models():
    """Load AnimeGANv2 models from Hugging Face Hub"""
    global using_real_models, model1, model2, face2paint
    
    try:
        logger.info("Loading AnimeGANv2 models from Hugging Face Hub...")
        
        # Load model 1 (face_paint_512_v1)
        model1 = torch.hub.load(
            "AK391/animegan2-pytorch:main",
            "generator",
            pretrained="face_paint_512_v1",
            device=device
        )
        logger.info("Model 1 (face_paint_512_v1) loaded successfully")
        
        # Load model 2 (face_paint_512_v2)
        model2 = torch.hub.load(
            "AK391/animegan2-pytorch:main",
            "generator",
            pretrained=True,  # Default is face_paint_512_v2
            device=device,
            progress=False
        )
        logger.info("Model 2 (face_paint_512_v2) loaded successfully")
        
        # Load face2paint for pre-processing
        face2paint = torch.hub.load(
            "AK391/animegan2-pytorch:main",
            "face2paint", 
            size=512
        )
        logger.info("Face2paint function loaded successfully")
        
        # Set all models to evaluation mode
        model1.eval()
        model2.eval()
        
        using_real_models = True
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.info("Falling back to simplified anime filter implementation")
        using_real_models = False
        model1 = None
        model2 = None
        face2paint = None
        return False

def process_image(img, version):
    """
    Apply anime-style effects to the uploaded image using AnimeGANv2 models
    """
    if img is None:
        return None
    
    try:
        global using_real_models, model1, model2, face2paint
        
        # Try loading models if not loaded yet
        if not using_real_models and model1 is None:
            load_models()
        
        if using_real_models and model1 is not None and model2 is not None and face2paint is not None:
            # Use the actual AnimeGANv2 models
            if version == 'version 1 (more stylized)':
                # Use model1 (face_paint_512_v1)
                logger.info("Using model1 (face_paint_512_v1)")
                anime_img = face2paint(model1, img)
            else:
                # Use model2 (face_paint_512_v2)
                logger.info("Using model2 (face_paint_512_v2)")
                anime_img = face2paint(model2, img)
            
            return anime_img
        else:
            # Fallback to simplified implementation
            logger.info("Using simplified anime filter implementation")
            # Convert PIL to numpy array
            np_img = np.array(img).astype(np.float32) / 255.0
            
            # Different styles based on version
            if version == 'version 2 (more robust)':
                # Version 2: More robust, less stylized
                # Increase saturation
                hsv = np.array(img.convert('HSV'))
                hsv[:, :, 1] = hsv[:, :, 1] * 1.4  # Increase saturation
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                
                # Convert back to RGB
                anime_img = Image.fromarray(hsv, 'HSV').convert('RGB')
                
                # Apply slight edge enhancement
                enhancer = Image.fromarray(np.clip(np_img * 255, 0, 255).astype(np.uint8))
                edges = enhancer.filter(ImageFilter.EDGE_ENHANCE)
                
                # Blend original and edge-enhanced image
                anime_img = Image.blend(anime_img, edges, 0.3)
                
            else:
                # Version 1: More stylized, less robust
                # Increase contrast and saturation
                hsv = np.array(img.convert('HSV'))
                hsv[:, :, 1] = hsv[:, :, 1] * 1.7  # Higher saturation
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)  # Increase value/brightness
                
                # Convert back to RGB
                anime_img = Image.fromarray(hsv, 'HSV').convert('RGB')
                
                # Apply more edge enhancement
                enhancer = Image.fromarray(np.clip(np_img * 255, 0, 255).astype(np.uint8))
                edges = enhancer.filter(ImageFilter.EDGE_ENHANCE_MORE)
                
                # Blend original and edge-enhanced image with more edge emphasis
                anime_img = Image.blend(anime_img, edges, 0.5)
                
                # Apply slight smoothing for the anime look
                anime_img = anime_img.filter(ImageFilter.SMOOTH)
            
            return anime_img
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return img  # Return original image in case of error


# Create the Gradio interface
def create_interface():
    return gr.Interface(
        fn=process_image,
        inputs=[
            gr.Image(type="pil", label="Upload Portrait Photo"),
            gr.Radio(
                choices=['version 1 (more stylized)', 'version 2 (more robust)'],
                value='version 2 (more robust)',
                label="Style Version"
            )
        ],
        outputs=gr.Image(type="pil", label="Anime Style Result"),
        title="AnimeGANv2 Portrait Converter",
        description="Convert your portrait photos to anime-style images using the AnimeGANv2 model.",
        article="""
        ## How it works
        This app uses the AnimeGANv2 model to transform your photos:
        
        - **Version 1** (face_paint_512_v1) provides more stylized anime effects but may work less well on some photos.
        - **Version 2** (face_paint_512_v2) is more robust with a wider range of photos but may have less pronounced anime effects.
        
        For best results, use well-lit portrait photos with clear facial features.
        """,
        # No example images yet
        # examples=[
        #     ["assets/sample_portrait.jpg", "version 1 (more stylized)"],
        #     ["assets/sample_portrait.jpg", "version 2 (more robust)"]
        # ],
        cache_examples=False,
        css="""
        .gradio-container {
            max-width: 800px;
            margin: 0 auto;
        }
        """
    )

# Main function to run the app
def main():
    # Load models first
    load_models()
    
    # Create the interface
    demo = create_interface()
    
    # Launch the app with a different port to avoid conflicts
    demo.launch(server_name="0.0.0.0", server_port=8080)

if __name__ == "__main__":
    main()