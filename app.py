import gradio as gr
from paddleocr import PaddleOCRVL

# Load model once
pipeline = PaddleOCRVL()

def run_ocr(image):

    result = pipeline.predict(image)

    text = ""

    for r in result:
        text += str(r) + "\n"

    return text


demo = gr.Interface(
    fn=run_ocr,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Handwriting OCR"
)

demo.launch()