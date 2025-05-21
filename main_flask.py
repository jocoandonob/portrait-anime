import os
from dotenv import load_dotenv
import io
import base64
import logging
import torch
import numpy as np
from PIL import Image, ImageFilter
from flask import Flask, render_template_string, request, redirect, url_for

# Load environment variables from .env file
load_dotenv()

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
        
        # Ensure the image is in RGB mode and properly sized
        if img.mode != 'RGB':
            logger.info(f"Converting input from {img.mode} to RGB")
            img = img.convert('RGB')
        
        # Resize image if too large (helps with performance and model constraints)
        max_size = 512  # Maximum size for any dimension
        if img.width > max_size or img.height > max_size:
            logger.info(f"Resizing image from {img.size} to fit within {max_size}x{max_size}")
            # Use default resampling which is available in all PIL versions
            img.thumbnail((max_size, max_size))
        
        if using_real_models and model1 is not None and model2 is not None and face2paint is not None:
            try:
                # Use the actual AnimeGANv2 models
                if version == 'version1':
                    # Use model1 (face_paint_512_v1)
                    logger.info("Using model1 (face_paint_512_v1)")
                    anime_img = face2paint(model1, img)
                else:
                    # Use model2 (face_paint_512_v2)
                    logger.info("Using model2 (face_paint_512_v2)")
                    anime_img = face2paint(model2, img)
                
                return anime_img
            except Exception as e:
                logger.error(f"Error using AnimeGAN models: {str(e)}")
                logger.info("Falling back to simplified implementation")
                # Continue to fallback implementation
        else:
            # Fallback to simplified implementation
            logger.info("Using simplified anime filter implementation")
            # Convert PIL to numpy array
            np_img = np.array(img).astype(np.float32) / 255.0
            
            # Different styles based on version
            if version == 'version2':
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

# Create Flask app
app = Flask(__name__)

# Load models at the start
load_models()

@app.route('/')
def index():
    """Main page with the upload form"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AnimeGANv2 Portrait Converter by Joco</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css" rel="stylesheet">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Montserrat:wght@400;600&display=swap');
            
            body {
                padding: 30px;
                max-width: 800px;
                margin: 0 auto;
                background-color: #121212;
                font-family: 'Montserrat', sans-serif;
            }
            
            .page-container {
                animation: fadeIn 1.2s ease-in-out;
            }
            
            .logo-container {
                text-align: center;
                margin-bottom: 20px;
            }
            
            .joco-logo {
                font-family: 'Playfair Display', serif;
                font-size: 24px;
                color: #e6c07b;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
                letter-spacing: 3px;
                border: 2px solid #e6c07b;
                display: inline-block;
                padding: 5px 15px;
                border-radius: 4px;
                margin-bottom: 20px;
                background: linear-gradient(145deg, #1a1a1a, #2a2a2a);
                transition: all 0.3s ease;
            }
            
            .joco-logo:hover {
                transform: scale(1.05);
                box-shadow: 0 0 15px rgba(230, 192, 123, 0.4);
            }
            
            .image-container {
                margin-top: 20px;
                text-align: center;
            }
            
            img {
                max-width: 100%;
                border-radius: 8px;
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            
            img:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
            }
            
            .form-container {
                background: linear-gradient(145deg, #1e1e1e, #2d2d2d);
                padding: 25px;
                border-radius: 8px;
                margin-top: 20px;
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                border: 1px solid #3a3a3a;
                animation: slideInUp 0.8s ease-out;
            }
            
            h1 {
                font-family: 'Playfair Display', serif;
                color: #e6c07b;
                margin-bottom: 20px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
                letter-spacing: 1px;
                text-align: center;
                font-weight: 700;
            }
            
            .description {
                margin-bottom: 30px;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
                text-align: center;
            }
            
            .lead {
                font-size: 1.2rem;
                line-height: 1.6;
            }
            
            .version-info {
                margin-top: 30px;
                background: linear-gradient(145deg, #1e1e1e, #2d2d2d);
                padding: 20px;
                border-radius: 8px;
                border: 1px solid #3a3a3a;
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                animation: slideInUp 1s ease-out;
            }
            
            .version-info h4 {
                color: #e6c07b;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
                border-bottom: 1px solid #3a3a3a;
                padding-bottom: 10px;
                margin-bottom: 15px;
            }
            
            .custom-btn {
                background: linear-gradient(145deg, #2e2e2e, #1a1a1a);
                color: #e6c07b;
                border: 1px solid #4a4a4a;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                transition: all 0.3s ease;
                font-weight: bold;
                letter-spacing: 1px;
                padding: 10px 20px;
            }
            
            .custom-btn:hover {
                background: linear-gradient(145deg, #3a3a3a, #2a2a2a);
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
                color: #fff;
            }
            
            .custom-radio .form-check-input:checked {
                background-color: #e6c07b;
                border-color: #e6c07b;
            }
            
            .form-control {
                background-color: #2c2c2c;
                border: 1px solid #4a4a4a;
                color: #eaeaea;
            }
            
            .form-control:focus {
                background-color: #333;
                border-color: #e6c07b;
                box-shadow: 0 0 0 0.25rem rgba(230, 192, 123, 0.25);
            }
            
            .footer {
                margin-top: 40px;
                text-align: center;
                font-size: 0.8rem;
                color: #888;
                animation: fadeIn 1.5s ease-in-out;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            
            @keyframes slideInUp {
                from {
                    transform: translateY(50px);
                    opacity: 0;
                }
                to {
                    transform: translateY(0);
                    opacity: 1;
                }
            }
            
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
            
            .pulse-animation {
                animation: pulse 2s infinite ease-in-out;
            }
        </style>
    </head>
    <body data-bs-theme="dark">
        <div class="container page-container">
            <div class="row">
                <div class="col-12">
                    <div class="logo-container animate__animated animate__fadeIn">
                        <div class="joco-logo pulse-animation">JOCO</div>
                    </div>
                    
                    <h1 class="animate__animated animate__fadeInDown">AnimeGANv2 Portrait Converter</h1>
                    <div class="description animate__animated animate__fadeIn">
                        <p class="lead">Convert your portrait photos to anime-style images using advanced machine learning models.</p>
                    </div>
                    
                    <div class="form-container">
                        <form action="/convert" method="post" enctype="multipart/form-data" class="mb-4">
                            <div class="mb-4">
                                <label for="image" class="form-label">Upload Portrait Photo</label>
                                <input type="file" class="form-control" id="image" name="image" accept="image/*" required>
                            </div>
                            
                            <div class="mb-4">
                                <label class="form-label">Style Version</label>
                                <div class="form-check custom-radio">
                                    <input class="form-check-input" type="radio" name="version" id="version1" value="version1">
                                    <label class="form-check-label" for="version1">
                                        Version 1 (more stylized)
                                    </label>
                                </div>
                                <div class="form-check custom-radio">
                                    <input class="form-check-input" type="radio" name="version" id="version2" value="version2" checked>
                                    <label class="form-check-label" for="version2">
                                        Version 2 (more robust)
                                    </label>
                                </div>
                            </div>
                            
                            <button type="submit" class="btn custom-btn w-100">Convert to Anime Style</button>
                        </form>
                    </div>
                    
                    <div class="version-info">
                        <h4>How it works</h4>
                        <p>This app uses the AnimeGANv2 model to transform your photos:</p>
                        <ul>
                            <li><strong>Version 1</strong> (face_paint_512_v1) provides more stylized anime effects but may work less well on some photos.</li>
                            <li><strong>Version 2</strong> (face_paint_512_v2) is more robust with a wider range of photos but may have less pronounced anime effects.</li>
                        </ul>
                        <p>For best results, use well-lit portrait photos with clear facial features.</p>
                    </div>
                    
                    <div class="footer">
                        <p>Created by Joco &copy; 2025 | Powered by AnimeGANv2</p>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/convert', methods=['POST'])
def convert():
    """Process the uploaded image and display the result"""
    if 'image' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))
    
    try:
        # Read the image data
        img_data = file.read()
        input_img = Image.open(io.BytesIO(img_data))
        
        # Convert RGBA images to RGB by removing alpha channel
        if input_img.mode == 'RGBA':
            logger.info("Converting RGBA image to RGB")
            # Create a white background
            background = Image.new('RGB', input_img.size, (255, 255, 255))
            # Paste the image on the background using alpha as mask
            background.paste(input_img, mask=input_img.split()[3])
            img = background
        else:
            # If already RGB, just convert to ensure RGB
            img = input_img.convert('RGB')
        
        # Get the style version
        version = request.form.get('version', 'version2')
        
        # Process the image
        result_img = process_image(img, version)
        
        if result_img is None:
            # If processing fails, return to index
            logger.error("Processing failed, result is None")
            return redirect(url_for('index'))
        
        # Ensure the result is in RGB mode for JPEG
        if result_img.mode != 'RGB':
            logger.info(f"Converting result from {result_img.mode} to RGB")
            result_img = result_img.convert('RGB')
        
        # Convert result image to base64 for displaying
        buffered = io.BytesIO()
        result_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Convert original image to base64 for displaying
        buffered_original = io.BytesIO()
        img.save(buffered_original, format="JPEG")
        img_str_original = base64.b64encode(buffered_original.getvalue()).decode('utf-8')
        
        # Display result
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AnimeGANv2 Portrait Converter by Joco - Result</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css" rel="stylesheet">
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Montserrat:wght@400;600&display=swap');
                
                body {
                    padding: 30px;
                    max-width: 900px;
                    margin: 0 auto;
                    background-color: #121212;
                    font-family: 'Montserrat', sans-serif;
                }
                
                .page-container {
                    animation: fadeIn 1.2s ease-in-out;
                }
                
                .logo-container {
                    text-align: center;
                    margin-bottom: 20px;
                }
                
                .joco-logo {
                    font-family: 'Playfair Display', serif;
                    font-size: 24px;
                    color: #e6c07b;
                    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
                    letter-spacing: 3px;
                    border: 2px solid #e6c07b;
                    display: inline-block;
                    padding: 5px 15px;
                    border-radius: 4px;
                    margin-bottom: 20px;
                    background: linear-gradient(145deg, #1a1a1a, #2a2a2a);
                    transition: all 0.3s ease;
                }
                
                .joco-logo:hover {
                    transform: scale(1.05);
                    box-shadow: 0 0 15px rgba(230, 192, 123, 0.4);
                }
                
                .image-container {
                    margin-top: 20px;
                    text-align: center;
                }
                
                img {
                    max-width: 100%;
                    border-radius: 8px;
                    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                    margin-bottom: 15px;
                }
                
                img:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
                }
                
                h1 {
                    font-family: 'Playfair Display', serif;
                    color: #e6c07b;
                    margin-bottom: 20px;
                    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
                    letter-spacing: 1px;
                    text-align: center;
                    font-weight: 700;
                }
                
                .description {
                    margin-bottom: 30px;
                    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
                    text-align: center;
                }
                
                .lead {
                    font-size: 1.2rem;
                    line-height: 1.6;
                }
                
                .comparison-container {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 30px;
                    justify-content: center;
                    animation: slideInUp 0.8s ease-out;
                }
                
                .image-card {
                    flex: 1;
                    min-width: 300px;
                    background: linear-gradient(145deg, #1e1e1e, #2d2d2d);
                    padding: 25px;
                    border-radius: 8px;
                    border: 1px solid #3a3a3a;
                    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }
                
                .image-card:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 12px 20px rgba(0, 0, 0, 0.3);
                }
                
                .image-card h4 {
                    color: #e6c07b;
                    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
                    text-align: center;
                    margin-bottom: 15px;
                    border-bottom: 1px solid #3a3a3a;
                    padding-bottom: 10px;
                    font-family: 'Playfair Display', serif;
                }
                
                .custom-btn {
                    background: linear-gradient(145deg, #2e2e2e, #1a1a1a);
                    color: #e6c07b;
                    border: 1px solid #4a4a4a;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                    transition: all 0.3s ease;
                    font-weight: bold;
                    letter-spacing: 1px;
                    padding: 10px 20px;
                }
                
                .custom-btn:hover {
                    background: linear-gradient(145deg, #3a3a3a, #2a2a2a);
                    transform: translateY(-2px);
                    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
                    color: #fff;
                }
                
                .footer {
                    margin-top: 40px;
                    text-align: center;
                    font-size: 0.8rem;
                    color: #888;
                    animation: fadeIn 1.5s ease-in-out;
                }
                
                @media (max-width: 768px) {
                    .image-card {
                        flex: 100%;
                    }
                }
                
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                
                @keyframes slideInUp {
                    from {
                        transform: translateY(50px);
                        opacity: 0;
                    }
                    to {
                        transform: translateY(0);
                        opacity: 1;
                    }
                }
                
                @keyframes pulse {
                    0% { transform: scale(1); }
                    50% { transform: scale(1.05); }
                    100% { transform: scale(1); }
                }
                
                .pulse-animation {
                    animation: pulse 2s infinite ease-in-out;
                }
                
                .animate-result {
                    animation: fadeIn 1s ease-in;
                }
            </style>
        </head>
        <body data-bs-theme="dark">
            <div class="container page-container">
                <div class="row">
                    <div class="col-12">
                        <div class="logo-container animate__animated animate__fadeIn">
                            <div class="joco-logo pulse-animation">JOCO</div>
                        </div>
                    
                        <h1 class="animate__animated animate__fadeInDown">AnimeGANv2 Portrait Converter</h1>
                        <div class="description animate__animated animate__fadeIn">
                            <p class="lead">Here's your anime-style transformation!</p>
                        </div>
                        
                        <div class="comparison-container">
                            <div class="image-card animate__animated animate__fadeInLeft">
                                <h4>Original Image</h4>
                                <img src="data:image/jpeg;base64,{{ original_img }}" alt="Original Image">
                            </div>
                            
                            <div class="image-card animate__animated animate__fadeInRight">
                                <h4>Anime Style Result</h4>
                                <img src="data:image/jpeg;base64,{{ result_img }}" alt="Anime Style Result">
                            </div>
                        </div>
                        
                        <div class="text-center mt-4 animate__animated animate__fadeInUp">
                            <a href="/" class="btn custom-btn">Convert Another Image</a>
                        </div>
                        
                        <div class="footer">
                            <p>Created by Joco &copy; 2025 | Powered by AnimeGANv2</p>
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        return render_template_string(
            html, 
            result_img=img_str, 
            original_img=img_str_original
        )
        
    except Exception as e:
        logger.error(f"Error in convert route: {str(e)}")
        return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)