# app.py
import gradio as gr
from paddleocr import PaddleOCRVL

# Initialize PaddleOCRVL pipeline
pipeline = PaddleOCRVL(
    pipeline_version="v1.5",
    use_layout_detection=True,
)

def run_ocr(image):
    # image can be a path or numpy array from Gradio
    output = pipeline.predict(image)
    return str(output)  # Convert to string to display easily

# Gradio interface
iface = gr.Interface(
    fn=run_ocr,
    inputs=gr.Image(type="filepath"),  # Gradio will pass the image path
    outputs="text",
    title="PaddleOCRVL on Hugging Face Spaces",
    description="Upload an image and get OCR output using PaddleOCRVL",
)

if __name__ == "__main__":
    iface.launch()