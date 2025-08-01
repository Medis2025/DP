import os
import fitz  # PyMuPDF
import cv2
import numpy as np
import layoutparser as lp
from tqdm import tqdm
from paddleocr import PaddleOCR

class PDFToMarkdown:
    def __init__(self, pdf_path, output_base_dir):
        self.pdf_path = pdf_path
        self.pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        self.output_base_dir = output_base_dir
        self.segment_output_dir = os.path.join(output_base_dir, self.pdf_name)
        self.text_output_dir = os.path.join(self.segment_output_dir, "text")
        self.txt_folder = os.path.join(self.segment_output_dir, "txt")
        self.img_folder = os.path.join(self.segment_output_dir, "sege", "img")
        self.text_raw_folder = os.path.join(self.segment_output_dir, "sege", "text_raw")
        self.markdown_path = os.path.join(self.segment_output_dir, f"{self.pdf_name}.md")

        for folder in [self.segment_output_dir, self.text_output_dir, self.txt_folder, self.img_folder, self.text_raw_folder]:
            os.makedirs(folder, exist_ok=True)

        label_map = {0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"}
        self.model = lp.Detectron2LayoutModel(
            config_path="/cluster/home/gw/detectron2/config.yml",
            model_path="/cluster/home/gw/detectron2/layout_weights.pth",
            label_map=label_map,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]
        )

        self.ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            use_textline_orientation=True,
            text_det_unclip_ratio=2.0,
            text_det_box_thresh=0.8,
            text_rec_score_thresh=0.8
        )

    def process(self):
        doc = fitz.open(self.pdf_path)
        all_markdown_lines = []
        first_title_used = False

        for page_index in tqdm(range(len(doc)), desc="Processing pages"):
            pix = doc[page_index].get_pixmap(dpi=150)
            image_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            layout = self.model.detect(image_np)
            sorted_blocks = sorted(layout, key=lambda b: (b.block.y_1, b.block.x_1))

            for block_id, block in enumerate(sorted_blocks):
                block_type = block.type.lower()
                offset = 10 if block_type == "title" else 5 if block_type in ["table", "figure"] else 3 if block_type == "text" else 0

                x1 = max(int(block.block.x_1) - offset, 0)
                y1 = max(int(block.block.y_1) - offset, 0)
                x2 = min(int(block.block.x_2) + offset, image_np.shape[1])
                y2 = min(int(block.block.y_2) + offset, image_np.shape[0])
                segment = image_np[y1:y2, x1:x2]

                if segment.size == 0:
                    continue

                padding = 20
                h, w, c = segment.shape
                canvas = np.ones((h + 2 * padding, w + 2 * padding, c), dtype=np.uint8) * 255
                canvas[padding:padding + h, padding:padding + w] = segment
                segment = canvas

                seg_name = f"page{page_index}_block{block_id}_{block.type}"
                seg_img_path = os.path.join(self.segment_output_dir, f"{seg_name}.png")
                text_path = os.path.join(self.text_output_dir, f"{seg_name}.txt")
                txt_path = os.path.join(self.txt_folder, f"{seg_name}.txt")
                img_output_path = os.path.join(self.img_folder, f"{seg_name}.png")
                text_raw_path = os.path.join(self.text_raw_folder, f"{seg_name}.txt")

                cv2.imwrite(seg_img_path, segment)

                try:
                    ocr_result = self.ocr.predict(segment)
                    texts = []
                    for blk in ocr_result:
                        for txt, score in zip(blk.get("rec_texts", []), blk.get("rec_scores", [])):
                            texts.append(txt)

                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(texts))
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(texts))

                    markdown_block = ""
                    if block_type == "title":
                        header_prefix = "#" if not first_title_used else "##"
                        markdown_block = f"{header_prefix} {' '.join(texts)}\n"
                        first_title_used = True
                        with open(text_raw_path, "w", encoding="utf-8") as f:
                            f.write("\n".join(texts))
                    elif block_type == "figure":
                        cv2.imwrite(img_output_path, segment)
                        rel_img_path = os.path.relpath(img_output_path, self.segment_output_dir)
                        markdown_block = f"![{seg_name}]({rel_img_path})\n"
                    else:
                        with open(text_raw_path, "w", encoding="utf-8") as f:
                            f.write("\n".join(texts))
                        markdown_block = f"{' '.join(texts)}\n"

                    all_markdown_lines.append(markdown_block)

                except Exception as e:
                    print(f"⚠️ OCR failed on {seg_name}: {e}")

        with open(self.markdown_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(all_markdown_lines))

        print(f"✅ All segment images saved to: {self.segment_output_dir}")
        print(f"✅ All OCR texts saved to: {self.text_output_dir}")
        print(f"✅ Sorted text files saved to: {self.txt_folder}")
        print(f"✅ Images saved to: {self.img_folder}")
        print(f"✅ Text raw files saved to: {self.text_raw_folder}")
        print(f"✅ Markdown file saved to: {self.markdown_path}")


if __name__ == "__main__":
    pdf_path = "/cluster/home/gw/MATRIX/regex_variants/segement/Matrix_reper/offset_out2/pdf_img/zcad030/zcad030.pdf"
    output_base_dir = "/cluster/home/gw/MATRIX/regex_variants/segement/Matrix_reper/test_output"
    converter = PDFToMarkdown(pdf_path, output_base_dir)
    converter.process()
