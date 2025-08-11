import os
import re
import yaml
import numpy as np
from tqdm import tqdm

class MatrixUtils:
    def __init__(self, 
                 grid_size=(3, 3), 
                 variant_yml_path=None,
                 image_root_dir=None,
                 base_pdf_dir=None,
                 output_matrix_path=None,
                 output_variant_path=None):
        """
        Parameters:
            grid_size: tuple(int, int), default (3, 3) - grid layout size (rows, cols)
            variant_yml_path: str - path to YAML file defining regex variant patterns
            image_root_dir: str - root directory where processed images are stored
            base_pdf_dir: str - base directory to search for PDF files
            output_matrix_path: str - output path to save layout_matrix.jsonl
            output_variant_path: str - output path to save variant_matrix.jsonl
        """
        self.grid_size = grid_size
        self.variant_yml_path = variant_yml_path
        self.image_root_dir = image_root_dir
        self.base_pdf_dir = base_pdf_dir
        self.output_matrix_path = output_matrix_path
        self.output_variant_path = output_variant_path

        self.variant_patterns = self.load_variant_patterns_from_yml(variant_yml_path) if variant_yml_path else []

    def get_matrix_position(self, bbox, image_shape):
        h, w = image_shape[:2]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        col = min(self.grid_size[1] - 1, int(cx * self.grid_size[1] / w))
        row = min(self.grid_size[0] - 1, int(cy * self.grid_size[0] / h))
        return row, col

    def is_already_processed(self, pdf_path):
        if not self.image_root_dir:
            raise ValueError("image_root_dir must be set to check processed files.")

        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_folder = os.path.join(self.image_root_dir, pdf_name)
        return os.path.exists(output_folder) and os.listdir(output_folder)

    def load_variant_patterns_from_yml(self, path):
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            return [re.compile(p, re.IGNORECASE) for p in config.get('variant_patterns_ignore_sep', [])]
        except Exception as e:
            print(f"⚠️ Failed to load variant patterns from {path}: {e}")
            return []

    def match_variants_in_text(self, text):
        matches = []
        for pattern in self.variant_patterns:
            for match in pattern.finditer(text):
                matches.append({
                    "variant": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "context": text[max(0, match.start() - 50):match.end() + 50],
                    "pattern": pattern.pattern
                })
        return matches

    def find_all_pdfs(self):
        if not self.base_pdf_dir:
            raise ValueError("base_pdf_dir must be provided.")

        pdf_paths = []
        for dirpath, _, filenames in os.walk(self.base_pdf_dir):
            for fname in filenames:
                if fname.lower().endswith(".pdf"):
                    full_path = os.path.join(dirpath, fname)
                    pdf_paths.append(full_path)
        return pdf_paths

    def batch_process_pdfs(self, processor_class):
        pdf_paths = self.find_all_pdfs()
        for pdf_path in tqdm(pdf_paths, desc="Batch Processing PDFs"):
            try:
                rel_path = os.path.relpath(pdf_path, self.base_pdf_dir)  # relative path from base
                pdf_key = rel_path.replace(os.sep, "_").replace(".pdf", "")  # flatten path to single name
                output_dir = os.path.join(self.image_root_dir, pdf_key)
                processor = processor_class(pdf_path, output_dir)
                processor.process()
            except Exception as e:
                print(f"❌ Failed to process {pdf_path}: {e}")
