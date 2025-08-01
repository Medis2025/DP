import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import fitz  # PyMuPDF
import cv2
import numpy as np
import layoutparser as lp
from tqdm import tqdm
from paddleocr import PaddleOCR
from collections import defaultdict
import re
import yaml
import matplotlib.pyplot as plt
import shutil

class SegmentProcessor:
    def __init__(self, config_path="/cluster/home/gw/MATRIX/regex_variants/segement/Matrix_reper/pdf_mat/sege/seg_config.yml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.padding = self.config.get("padding", 20)
        self.offsets = self.config.get("offsets", {
            "title": 10,
            "table": 5,
            "figure": 5,
            "text": 3
        })
        self.ocr_config = self.config.get("ocr", {})
        self.ocr = PaddleOCR(**self.ocr_config)

    def apply_offset_and_padding(self, block, image_np):
        block_type = block.type.lower()
        offset = self.offsets.get(block_type, 0)

        x1 = max(int(block.block.x_1) - offset, 0)
        y1 = max(int(block.block.y_1) - offset, 0)
        x2 = min(int(block.block.x_2) + offset, image_np.shape[1])
        y2 = min(int(block.block.y_2) + offset, image_np.shape[0])
        segment = image_np[y1:y2, x1:x2]

        if segment.size == 0:
            return None, x1, y1

        h, w, c = segment.shape
        canvas = np.ones((h + 2 * self.padding, w + 2 * self.padding, c), dtype=np.uint8) * 255
        canvas[self.padding:self.padding + h, self.padding:self.padding + w] = segment
        return canvas, x1 - self.padding, y1 - self.padding