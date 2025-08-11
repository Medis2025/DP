import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use CUDA:1

import fitz  # PyMuPDF
import cv2
import numpy as np
import layoutparser as lp
from tqdm import tqdm
from sege.block_sorter import BlockSorter
from sege.doctr_drop_du import BlockProcessor
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from PIL import Image
import torch
import yaml
import concurrent.futures
from ocr_model.doctrocr import DoctrOCR


class PDFToMarkdown:
    def __init__(self, pdf_path, output_base_dir):
        self.pdf_path = pdf_path
        self.pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        self.output_base_dir = output_base_dir
        self.segment_output_dir = os.path.join(output_base_dir, self.pdf_name)
        self.text_output_dir = os.path.join(self.segment_output_dir, "text")
        self.txt_folder = os.path.join(self.segment_output_dir, "txt")
        self.img_folder = os.path.join(self.segment_output_dir, "sege", "img")
        self.text_raw_folder = os.path.join(self.segment_output_dir, "text_raw")
        self.markdown_path = os.path.join(self.segment_output_dir, f"{self.pdf_name}.md")
        self.config_path = "/cluster/home/gw/MATRIX/regex_variants/segement/Matrix_reper/pdf_mat/sege/seg_config.yml"

        for folder in [self.segment_output_dir, self.text_output_dir, self.txt_folder, self.img_folder, self.text_raw_folder]:
            os.makedirs(folder, exist_ok=True)

        # Load unified YAML config
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        detectron_config = config.get("detectron", {})
        ocr_config = config.get("dococr", {})
        doctr_config_path = self.config_path

        label_map = detectron_config.get("label_map", {0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"})

        self.model = lp.Detectron2LayoutModel(
            config_path=detectron_config["config_path"],
            model_path=detectron_config["model_path"],
            label_map=label_map,
            extra_config=detectron_config.get("extra_config", ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.3]),
            device=detectron_config.get("device", "cuda")
        )

        self.ocr_engine = DoctrOCR(
            det_arch=ocr_config.get("det_arch", 'db_resnet50'),
            reco_arch=ocr_config.get("reco_arch", 'crnn_vgg16_bn'),
            conf_threshold=ocr_config.get("conf_threshold", 0.3),
            add_score=ocr_config.get("add_score", False)
        )

        self.block_processor = BlockProcessor(
            self.segment_output_dir,
            self.text_output_dir,
            self.txt_folder,
            self.img_folder,
            self.text_raw_folder,
            self.ocr_engine,
            config_path=doctr_config_path
        )

    def process(self):
        doc = fitz.open(self.pdf_path)
        all_markdown_lines = []
        first_title_used = False

        for page_index in tqdm(range(len(doc)), desc="Processing pages"):
            pix = doc[page_index].get_pixmap(dpi=450)
            image_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

            sorter = BlockSorter(image_np.shape[1])
            raw_blocks = self.model.detect(image_np)
            sorted_blocks = sorter.sort_blocks(raw_blocks)

            filtered_blocks = self.block_processor.filter_overlapping_blocks(sorted_blocks)
            print(f"Filtered {len(sorted_blocks) - len(filtered_blocks)} overlapping blocks on page {page_index}.")

            for block_id, block in enumerate(filtered_blocks):
                args = (block, page_index, block_id, image_np, first_title_used)
                markdown_block = self.block_processor.process_block(args)
                all_markdown_lines.append(markdown_block)

        with open(self.markdown_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(all_markdown_lines))

        print(f"✅ All segment images saved to: {self.segment_output_dir}")
        print(f"✅ All OCR texts saved to: {self.text_output_dir}")
        print(f"✅ Sorted text files saved to: {self.txt_folder}")
        print(f"✅ Images saved to: {self.img_folder}")
        print(f"✅ Text raw files saved to: {self.text_raw_folder}")
        print(f"✅ Markdown file saved to: {self.markdown_path}")


if __name__ == "__main__":
    pdf_path = "/cluster/home/gw/MATRIX/regex_variants/segement/Matrix_reper/pdf_mat/test_data/Genetics_in_Medicine/Genetics in Medicine 13.pdf"
    output_base_dir = "/cluster/home/gw/MATRIX/regex_variants/segement/Matrix_reper/test_output"
    converter = PDFToMarkdown(pdf_path, output_base_dir)
    converter.process()
