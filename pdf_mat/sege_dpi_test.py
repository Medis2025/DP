import os
import fitz  # PyMuPDF
import cv2
import numpy as np
import layoutparser as lp
from tqdm import tqdm
from paddleocr import PaddleOCR

# === Input PDF path ===
pdf_path = "/cluster/home/gw/MATRIX/regex_variants/segement/Matrix_reper/offset_out2/pdf_img/zcad030/zcad030.pdf"
pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

# === Output base directory ===
output_base_dir = "/cluster/home/gw/MATRIX/regex_variants/segement/Matrix_reper/test_output"
segment_output_dir = os.path.join(output_base_dir, pdf_name)
text_output_dir = os.path.join(segment_output_dir, "text")
os.makedirs(segment_output_dir, exist_ok=True)
os.makedirs(text_output_dir, exist_ok=True)

# === Load layout detection model ===
label_map = {0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"}
model = lp.Detectron2LayoutModel(
    config_path="/cluster/home/gw/detectron2/config.yml",
    model_path="/cluster/home/gw/detectron2/layout_weights.pth",
    label_map=label_map,
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]
)

# === Load PaddleOCR with document-specific preprocessing ===
ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_textline_orientation=True,
    text_det_unclip_ratio=2.0,
    text_det_box_thresh=0.8,
    text_rec_score_thresh=0.8
)

# === Load PDF ===
doc = fitz.open(pdf_path)

# === Process each page ===
for page_index in tqdm(range(len(doc)), desc="Processing pages"):
    pix = doc[page_index].get_pixmap(dpi=150)
    image_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    layout = model.detect(image_np)

    for block_id, block in enumerate(layout):
        # Set custom offset based on block type
        block_type = block.type.lower()
        if block_type == "title":
            offset = 10
        elif block_type in ["table", "figure"]:
            offset = 5
        elif block_type == "text":
            offset = 3
        else:
            offset = 0

        x1 = max(int(block.block.x_1) - offset, 0)
        y1 = max(int(block.block.y_1) - offset, 0)
        x2 = min(int(block.block.x_2) + offset, image_np.shape[1])
        y2 = min(int(block.block.y_2) + offset, image_np.shape[0])
        segment = image_np[y1:y2, x1:x2]

        if segment.size == 0:
            continue

        # Add white padding
        padding = 20
        h, w, c = segment.shape
        canvas = np.ones((h + 2 * padding, w + 2 * padding, c), dtype=np.uint8) * 255
        canvas[padding:padding + h, padding:padding + w] = segment
        segment = canvas

        seg_name = f"page{page_index}_block{block_id}_{block.type}"
        seg_img_path = os.path.join(segment_output_dir, f"{seg_name}.png")
        text_path = os.path.join(text_output_dir, f"{seg_name}.txt")

        # Save segment image
        cv2.imwrite(seg_img_path, segment)

        # Run OCR and save text
        try:
            ocr_result = ocr.predict(segment)
            texts = []
            for block in ocr_result:
                print(ocr_result)
                for txt, score in zip(block.get("rec_texts", []), block.get("rec_scores", [])):
                    texts.append(f"{txt} (score={score:.2f})")

            with open(text_path, "w", encoding="utf-8") as f:
                f.write("\n".join(texts))
        except Exception as e:
            print(f"\u26a0\ufe0f OCR failed on {seg_name}: {e}")

print(f"✅ All segment images saved to: {segment_output_dir}")
print(f"✅ All OCR texts saved to: {text_output_dir}")
