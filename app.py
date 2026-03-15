import gradio as gr
from paddleocr import PaddleOCRVL
from PIL import Image

# Initialize PaddleOCRVL pipeline
pipeline = PaddleOCRVL(
    pipeline_version="v1.5",
    use_layout_detection=True,
)

def run_ocr(image):
    """
    Gradio passes either a filepath (str) or a PIL.Image
    """
    # If Gradio gives a filepath, open it with PIL
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    # Run OCR
    output = pipeline.predict(image)
    
    # Save results
    for res in output:
        res.save_to_json(save_path="output")
        res.save_to_markdown(save_path="output")
    
    # Return a concise string summary for display
    results_text = "\n".join([res.text for res in output])
    return results_text

# Gradio interface
iface = gr.Interface(
    fn=run_ocr,
    inputs=gr.Image(type="pil"),  # Use PIL image to avoid numpy scalar issues
    outputs="text",
    title="PaddleOCRVL on Hugging Face Spaces",
    description="Upload an image and get OCR output using PaddleOCRVL",
)

if __name__ == "__main__":
    iface.launch()