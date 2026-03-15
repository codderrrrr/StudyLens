import gradio as gr
import numpy as np
from paddleocr import PaddleOCRVL

# Initialize PaddleOCRVL
pipeline = PaddleOCRVL(
    pipeline_version="v1.5",
    use_layout_detection=True,
)

def run_ocr(image):
    """
    image: PIL.Image from Gradio
    """
    # Convert PIL image to NumPy array
    img_np = np.array(image)
    output = pipeline.predict(img_np)
    
    # Convert output to string for display
    return str(output)

# Gradio interface
iface = gr.Interface(
    fn=run_ocr,
    inputs=gr.Image(type="pil"),  # Pass PIL image from Gradio
    outputs="text",
    title="PaddleOCRVL on Hugging Face Spaces",
    description="Upload an image and get OCR output using PaddleOCRVL",
)

if __name__ == "__main__":
    iface.launch()