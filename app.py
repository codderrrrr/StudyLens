import gradio as gr
import numpy as np
from paddleocr import PaddleOCR, PPStructure

# -------------------
# Load models ONCE
# -------------------
ocr = PaddleOCR(use_angle_cls=True, lang="en", ocr_version="PP-OCRv5")
structure = PPStructure(structure_version="PP-StructureV3", show_log=False)

# -------------------
# OCR function
# -------------------
def run_ocr(image):
    img_np = np.array(image)

    # -------- text OCR --------
    ocr_result = ocr.ocr(img_np, cls=True)

    text_output = []
    for line in ocr_result[0]:
        text_output.append(line[1][0])

    # -------- structure (optional) --------
    structure_result = structure(img_np)

    return {
        "text": "\n".join(text_output),
        "structure": str(structure_result)
    }

# -------------------
# UI
# -------------------
iface = gr.Interface(
    fn=run_ocr,
    inputs=gr.Image(type="pil"),
    outputs="json",
    title="PaddleOCR v5 + PP-StructureV3",
    description="Fast CPU OCR + document structure extraction"
)

if __name__ == "__main__":
    iface.launch()