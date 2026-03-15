import gradio as gr
import numpy as np
from paddleocr import PaddleOCRVL

# Load model once
pipeline = PaddleOCRVL()

def run_ocr(image):
    # Convert PIL image to numpy array
    image_np = np.array(image)

    # Run OCR
    result = pipeline.predict(image_np)

    # Format result
    text = ""
    for r in result:
        text += str(r) + "\n"

    return text

demo = gr.Interface(
    fn=run_ocr,
    inputs=gr.Image(type="pil"),  # still receives PIL from Gradio
    outputs="text",
    title="Handwriting OCR"
)

demo.launch()