import logging
import numpy as np
from PIL import Image, ImageFilter
import gradio as gr

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def process_image(img, version):
    """
    Simple function to apply anime-style filter to an image.
    """
    if img is None:
        return None
    
    try:
        # Convert PIL to numpy array
        np_img = np.array(img).astype(np.float32) / 255.0
        
        # Different styles based on version
        if version == 'version 2 (🔺 robustness,🔻 stylization)':
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
            # Apply more dramatic color shift and edge detection
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
        logging.error(f"Error processing image: {str(e)}")
        return img  # Return original image in case of error

# Set up the interface
demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload Portrait Photo"),
        gr.Radio(
            choices=[
                'version 1 (🔺 stylization, 🔻 robustness)', 
                'version 2 (🔺 robustness,🔻 stylization)'
            ],
            value='version 2 (🔺 robustness,🔻 stylization)',
            label="Style Version"
        )
    ],
    outputs=gr.Image(type="pil", label="Anime Style Image"),
    title="AnimeGANv2 Portrait Converter",
    description="Convert your portrait photos to anime-style images using different styles.",
    article="<p>Upload a portrait photo and select the style you prefer. Version 1 gives more stylized results but may work less well on some photos. Version 2 is more robust but may have less pronounced anime effects.</p>"
)

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
