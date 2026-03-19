import gradio as gr
import numpy as np
from paddleocr import PaddleOCRVL
import asyncio

pipeline = PaddleOCRVL(
    pipeline_version="v1.5",
    use_layout_detection=True,
)

async def run_ocr(image):
    img_np = np.array(image)
    output = await asyncio.to_thread(
        pipeline.predict, img_np
    )
    return str(output)

iface = gr.Interface(
    fn=run_ocr,
    inputs=gr.Image(type="pil"),  
    outputs="text",
    title="PaddleOCRVL on Hugging Face Spaces",
    description="Upload an image and get OCR output using PaddleOCRVL",
)

if __name__ == "__main__":
    iface.launch()