import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use CUDA:1
import fitz  # PyMuPDF
import cv2
import numpy as np
import layoutparser as lp
from tqdm import tqdm
from sege.block_sorter import BlockSorter
from sege.doctr_markdown import BlockProcessor
from PIL import Image
import yaml
import json

from matrix.MatrixUtils import MatrixUtils
from ocr_model.doctr_box import DoctrOCR
from matrix.variant_extractor import VariantExtractor
from pdf_find.pdf_finder import PDFFinder

class PDFToMarkdown:
    def __init__(self, pdf_path, output_base_dir):
        self.pdf_path = pdf_path
        self.pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        self.output_base_dir = output_base_dir

        self.segment_output_dir = os.path.join(output_base_dir, "pdf_img", self.pdf_name)
        self.raw_dir = os.path.join(self.segment_output_dir, "raw")
        self.overlay_dir = os.path.join(self.segment_output_dir, "overlay")
        self.text_dir = os.path.join(self.segment_output_dir, "text")
        self.matrix_dir = os.path.join(self.segment_output_dir, "matrix")
        self.txt_folder = os.path.join(self.segment_output_dir, "txt")
        self.img_folder = os.path.join(self.segment_output_dir, "sege", "img")
        self.text_raw_folder = os.path.join(self.segment_output_dir, "text_raw")
        self.markdown_path = os.path.join(self.segment_output_dir, f"{self.pdf_name}.md")
        self.config_path = "/cluster/home/gw/MATRIX/regex_variants/segement/Matrix_reper/pdf_mat/sege/seg_config.yml"

        self.prepare_output_dirs()

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

        detectron_config = config.get("detectron", {})
        ocr_config = config.get("dococr", {})
        matrix_cfg = config.get("matrix", {})

        label_map = detectron_config.get(
            "label_map", {0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"}
        )
        score_thresh = float(detectron_config.get("score_thresh", 0.3))

        self.model = lp.Detectron2LayoutModel(
            config_path=detectron_config["config_path"],
            model_path=detectron_config["model_path"],
            label_map=label_map,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", score_thresh],
            device=detectron_config.get("device", "cuda"),
        )

        self.ocr_engine = DoctrOCR(
            det_arch=ocr_config.get("det_arch", 'db_resnet50'),
            reco_arch=ocr_config.get("reco_arch", 'crnn_vgg16_bn'),
            conf_threshold=float(ocr_config.get("conf_threshold", 0.3)),
            add_score=bool(ocr_config.get("add_score", False)),
        )

        self.block_processor = BlockProcessor(
            self.segment_output_dir,
            self.text_dir,
            self.txt_folder,
            self.img_folder,
            self.text_raw_folder,
            self.ocr_engine,
            config_path=self.config_path,
        )

        grid_h = int(matrix_cfg.get("grid_h", 3))
        grid_w = int(matrix_cfg.get("grid_w", 3))
        self.matrix_utils = MatrixUtils(grid_size=(grid_h, grid_w))
        self.variant_extractor = VariantExtractor(
            variant_yml_path="/cluster/home/gw/MATRIX/regex_variants/segement/patterns/variant_patterns_ignore_sep.yml",
            ocr_engine=self.ocr_engine,
            segment_processor=self.block_processor,
            get_matrix_position_fn=self.matrix_utils.get_matrix_position,
        )

    def prepare_output_dirs(self):
        for folder in [
            self.segment_output_dir,
            self.raw_dir,
            self.overlay_dir,
            self.text_dir,
            self.matrix_dir,
            self.txt_folder,
            self.img_folder,
            self.text_raw_folder,
        ]:
            os.makedirs(folder, exist_ok=True)

    @staticmethod
    def _pixmap_to_rgb_numpy(pix: fitz.Pixmap) -> np.ndarray:
        if pix.n >= 4:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        arr = arr.reshape(pix.h, pix.w, 3)
        return arr

    def process(self):
        doc = fitz.open(self.pdf_path)
        all_markdown_lines = []
        first_title_used = False

        for page_index in tqdm(range(len(doc)), desc=f"Processing pages for {self.pdf_name}"):
            page = doc[page_index]
            pix = page.get_pixmap(dpi=450)
            image_rgb = self._pixmap_to_rgb_numpy(pix)

            raw_out = os.path.join(self.raw_dir, f"page{page_index}.png")
            Image.fromarray(image_rgb).save(raw_out)

            sorter = BlockSorter(image_rgb.shape[1])
            raw_blocks = self.model.detect(image_rgb)
            sorted_blocks = sorter.sort_blocks(raw_blocks)

            filtered_blocks = self.block_processor.filter_overlapping_blocks(sorted_blocks)

            for block_id, block in enumerate(filtered_blocks):
                try:
                    md_block, first_title_used, _ = self.block_processor.process_block(
                        block, page_index, block_id, image_rgb, first_title_used
                    )
                    if md_block:
                        all_markdown_lines.append(md_block)
                except Exception as e:
                    print(f"⚠️ Block processing failed on page{page_index}_block{block_id}: {e}")

            try:
                matrix, variant_matrix = self.variant_extractor.extract_and_plot_variants(
                    image_np=image_rgb,
                    layout=filtered_blocks,
                    page_index=page_index,
                    pdf_name=self.pdf_name,
                    pdf_dir=self.segment_output_dir,
                )
                with open(os.path.join(self.matrix_dir, f"page{page_index}_layout.json"), 'w', encoding='utf-8') as f:
                    json.dump({"page": page_index, "matrix": matrix}, f, ensure_ascii=False, indent=2)
                with open(os.path.join(self.matrix_dir, f"page{page_index}_variant.json"), 'w', encoding='utf-8') as f:
                    json.dump({"page": page_index, "matrix": variant_matrix}, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"⚠️ Failed variant extraction on page {page_index}: {e}")

        try:
            with open(self.markdown_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(all_markdown_lines))
            print(f"✅ Markdown saved: {self.markdown_path}")
        except Exception as e:
            print(f"⚠️ Failed to save Markdown: {e}")

        print(f"✅ All outputs saved to: {self.segment_output_dir}")

if __name__ == "__main__":
    pdf_base_dir = "/cluster/home/gw/data"
    output_base_dir = "/cluster/home/gw/MATRIX/regex_variants/segement/Matrix_reper/test_output"

    finder = PDFFinder(pdf_base_dir, os.path.join(output_base_dir, "pdf_img"))
    all_pdfs = finder.find_pdfs()
    unprocessed_pdfs = finder.get_unprocessed_pdfs()

    total_pdfs = len(all_pdfs)
    print(f"Total PDFs in base dir: {total_pdfs}")
    print(f"Found {len(unprocessed_pdfs)} unprocessed PDFs")

    processed_count = 0

    for pdf_path in tqdm(unprocessed_pdfs, desc="Processing all PDFs", total=len(unprocessed_pdfs)):
        print(f"Processing PDF: {pdf_path}")
        converter = PDFToMarkdown(pdf_path, output_base_dir)
        converter.process()
        processed_count += 1

    skipped_count = total_pdfs - processed_count
    print(f"Processed {processed_count} PDFs. Skipped {skipped_count} already processed.")
