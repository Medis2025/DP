import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import fitz  # PyMuPDF
import layoutparser as lp
import cv2
from tqdm import tqdm
import numpy as np
from paddleocr import PaddleOCR
from collections import defaultdict
import re
import yaml
import matplotlib.pyplot as plt
import shutil

class LayoutGridExporter:
    def __init__(self, base_dir, output_dir, use_gpu=False, gpu_id=0, lang='en'):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.image_root_dir = os.path.join(output_dir, "pdf_img")
        self.plot_raw_dir = "raw"
        self.plot_overlay_dir = "overlay"
        self.plot_text_dir = "text"
        self.plot_matrix_dir = "matrix"

        self.matrix_output_path = os.path.join(output_dir, "layout_matrix.jsonl")
        self.variant_output_path = os.path.join(output_dir, "variant_matrix.jsonl")

        os.makedirs(self.output_dir, exist_ok=True)
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.label_map = {0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"}
        self.ocr = PaddleOCR(lang=lang, use_angle_cls=True)

    def get_matrix_position(self, bbox, image_shape, grid_size=(3, 3)):
        h, w = image_shape[:2]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        col = min(grid_size[1] - 1, int(cx * grid_size[1] / w))
        row = min(grid_size[0] - 1, int(cy * grid_size[0] / h))
        return row, col

    def is_already_processed(self, pdf_path):
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_folder = os.path.join(self.image_root_dir, pdf_name)
        return os.path.exists(output_folder) and os.listdir(output_folder)

    def load_variant_patterns_from_yml(self, path):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return [re.compile(p, re.IGNORECASE) for p in config['variant_patterns_ignore_sep']]

    def match_variants_in_text(self, text):
        matches = []
        variant_patterns_ignore_sep = self.load_variant_patterns_from_yml('/cluster/home/gw/MATRIX/regex_variants/segement/patterns/variant_patterns_ignore_sep.yml')
        for pattern in variant_patterns_ignore_sep:
            for match in pattern.finditer(text):
                matches.append({
                    "variant": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "context": text[max(0, match.start()-50):match.end()+50],
                    "pattern": pattern.pattern
                })
        return matches

    def extract_and_plot_variants(self, image_np, layout, page_index, pdf_name, pdf_dir):
        matrix = defaultdict(list)
        variant_matrix = defaultdict(list)
        fig, ax = plt.subplots(figsize=(12, 16))
        ax.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

        block_color_map = {
            "text": (0.8, 0.9, 1.0, 0.3),
            "title": (1.0, 0.8, 0.8, 0.3),
            "table": (0.8, 1.0, 0.8, 0.3),
            "figure": (0.9, 0.9, 0.9, 0.3)
        }

        text_lines = []

        for block in layout:
            x1, y1, x2, y2 = int(block.block.x_1), int(block.block.y_1), int(block.block.x_2), int(block.block.y_2)
            label = block.type
            row, col = self.get_matrix_position([x1, y1, x2, y2], image_np.shape)
            bbox = (x1, y1, x2, y2)
            matrix[(row, col)].append({"type": label.upper(), "bbox": bbox})
            segment = image_np[y1:y2, x1:x2]
            if segment.size == 0:
                continue
            matched_block = False
            try:
                ocr_result = self.ocr.predict(segment)
                for ocr_dict in ocr_result:
                    texts = ocr_dict.get("rec_texts", [])
                    boxes = ocr_dict.get("rec_boxes", [])
                    scores = ocr_dict.get("rec_scores", [])
                    for text, box, score in zip(texts, boxes, scores):
                        text_lines.append(f"[{row},{col}] {text} (score={score:.2f})")
                        variant_matches = self.match_variants_in_text(text)
                        for m in variant_matches:
                            rel_box = list(map(int, box))
                            abs_box = [rel_box[0] + x1, rel_box[1] + y1, rel_box[2] + x1, rel_box[3] + y1]
                            variant_matrix[(row, col)].append({
                                "variant": m["variant"],
                                "bbox": abs_box,
                                "type": label.upper(),
                                "context": m["context"],
                                "pattern": m["pattern"]
                            })
                            ax.add_patch(plt.Rectangle(
                                (abs_box[0], abs_box[1]),
                                abs_box[2] - abs_box[0],
                                abs_box[3] - abs_box[1],
                                fill=True, facecolor=(1.0, 1.0, 0.0, 0.5), edgecolor='orange', linewidth=2))
                            ax.text(abs_box[2] + 5, abs_box[1], m["variant"], fontsize=9, color='black',
                                    bbox=dict(facecolor='yellow', alpha=0.8, boxstyle='round'))
                            matched_block = True
            except Exception as e:
                print(f"⚠️ OCR or variant matching failed: {e}")

            if matched_block and label in block_color_map:
                ax.add_patch(plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=True, facecolor=block_color_map[label], edgecolor='blue', linewidth=1))

        os.makedirs(os.path.join(pdf_dir, self.plot_raw_dir), exist_ok=True)
        os.makedirs(os.path.join(pdf_dir, self.plot_overlay_dir), exist_ok=True)
        os.makedirs(os.path.join(pdf_dir, self.plot_text_dir), exist_ok=True)
        os.makedirs(os.path.join(pdf_dir, self.plot_matrix_dir), exist_ok=True)

        raw_img_path = os.path.join(pdf_dir, self.plot_raw_dir, f"page{page_index}.png")
        plot_img_path = os.path.join(pdf_dir, self.plot_overlay_dir, f"page{page_index}_plotted.png")
        text_path = os.path.join(pdf_dir, self.plot_text_dir, f"page{page_index}.txt")

        cv2.imwrite(raw_img_path, image_np)
        fig.savefig(plot_img_path)
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(text_lines))
        plt.close(fig)

        return matrix, variant_matrix

    def process_single_pdf(self, pdf_path):
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_dir = os.path.join(self.image_root_dir, pdf_name)
        os.makedirs(pdf_dir, exist_ok=True)

        if self.is_already_processed(pdf_path):
            print(f"⏭️ Skipping already processed: {pdf_path}")
            return [], []

        try:
            shutil.copy2(pdf_path, os.path.join(pdf_dir, os.path.basename(pdf_path)))
        except Exception as e:
            print(f"⚠️ Failed to copy PDF {pdf_path}: {e}")

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"❌ MuPDF error, skipping file {pdf_path}: {e}")
            return [], []

        model = lp.Detectron2LayoutModel(
            config_path="/cluster/home/gw/detectron2/config.yml",
            model_path="/cluster/home/gw/detectron2/layout_weights.pth",
            label_map=self.label_map,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8])

        layout_pages = []
        variant_pages = []

        for page_index in range(len(doc)):
            pix = doc[page_index].get_pixmap(dpi=150)
            image_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            layout = model.detect(image_np)

            matrix, variant_matrix = self.extract_and_plot_variants(image_np, layout, page_index, pdf_name, pdf_dir)

            layout_entry = {
                "page": page_index,
                "file": pdf_path,
                "matrix": {f"B[{i},{j}]": matrix.get((i, j), "<empty>") for i in range(3) for j in range(3)}
            }
            variant_entry = {
                "page": page_index,
                "file": pdf_path,
                "matrix": {f"B[{i},{j}]": variant_matrix.get((i, j), "<empty>") for i in range(3) for j in range(3)}
            }

            layout_pages.append(layout_entry)
            variant_pages.append(variant_entry)

            with open(os.path.join(pdf_dir, self.plot_matrix_dir, f"page{page_index}_layout.json"), 'w', encoding='utf-8') as f:
                json.dump(layout_entry, f, ensure_ascii=False, indent=2)
            with open(os.path.join(pdf_dir, self.plot_matrix_dir, f"page{page_index}_variant.json"), 'w', encoding='utf-8') as f:
                json.dump(variant_entry, f, ensure_ascii=False, indent=2)

        return layout_pages, variant_pages

    def process_all(self):
        pdf_files = []
        for dirpath, _, filenames in os.walk(self.base_dir):
            for file in filenames:
                if file.endswith(".pdf"):
                    pdf_files.append(os.path.join(dirpath, file))

        with open(self.matrix_output_path, 'a', encoding='utf-8') as layout_out, \
             open(self.variant_output_path, 'a', encoding='utf-8') as variant_out:

            for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
                try:
                    layout_pages, variant_pages = self.process_single_pdf(pdf_path)
                    for entry in layout_pages:
                        json.dump(entry, layout_out, ensure_ascii=False)
                        layout_out.write("\n")
                        layout_out.flush()
                    for entry in variant_pages:
                        json.dump(entry, variant_out, ensure_ascii=False)
                        variant_out.write("\n")
                        variant_out.flush()
                except Exception as e:
                    print(f"❌ Error processing {pdf_path}: {e}")

        print(f"✅ Layout matrix saved to: {self.matrix_output_path}")
        print(f"✅ Variant matrix saved to: {self.variant_output_path}")

if __name__ == "__main__":
    exporter = LayoutGridExporter(
        base_dir="/cluster/home/gw/data",
        output_dir="/cluster/home/gw/MATRIX/regex_variants/segement/Matrix_reper/output_mat",
        use_gpu=True,
        gpu_id=1)
    exporter.process_all()
