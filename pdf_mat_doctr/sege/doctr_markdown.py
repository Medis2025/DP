import os
import cv2
import json
import yaml
import hashlib
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

class BlockProcessor:
    """
    Block processor with:
      - OCR cache persisted to disk
      - OCR performed on in-memory numpy arrays (not file paths)
      - Public `get_ocr_payload(...)` for consumers (e.g., VariantExtractor)
      - Overlap filtering controlled by YAML
      - Per-segment artifacts written under `<segment_output_dir>/blocks`
    """

    def __init__(self, segment_output_dir, text_output_dir, txt_folder, img_folder, text_raw_folder, ocr_engine, config_path):
        self.segment_output_dir = segment_output_dir
        self.text_output_dir = text_output_dir
        self.txt_folder = txt_folder
        self.img_folder = img_folder
        self.text_raw_folder = text_raw_folder
        self.ocr_engine = ocr_engine

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        block_config = config.get('Block', {})

        self.padding = int(block_config.get('padding', 200))
        self.offsets = block_config.get('offsets', {'title': 10, 'table': 5, 'figure': 5, 'text': 3})
        self.overlap_threshold = float(block_config.get('overlap_threshold', 0.2))

        # Directories
        self.ocr_cache_dir = os.path.join(self.segment_output_dir, 'ocr_cache')
        self.ocr_cache_path = os.path.join(self.ocr_cache_dir, 'ocr_cache.json')
        self.segment_block_dir = os.path.join(self.segment_output_dir, 'blocks')
        for d in [self.segment_output_dir, self.text_output_dir, self.txt_folder, self.img_folder, self.text_raw_folder, self.ocr_cache_dir, self.segment_block_dir]:
            os.makedirs(d, exist_ok=True)

        # OCR cache (in-memory + persisted)
        self.ocr_cache: Dict[str, Dict[str, Any]] = {}
        self._load_ocr_cache()

    # ---------------- Cache helpers ----------------
    def _load_ocr_cache(self) -> None:
        if os.path.isfile(self.ocr_cache_path):
            try:
                with open(self.ocr_cache_path, 'r', encoding='utf-8') as f:
                    self.ocr_cache = json.load(f)
            except Exception:
                self.ocr_cache = {}

    def _flush_ocr_cache(self) -> None:
        tmp_path = self.ocr_cache_path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(self.ocr_cache, f, ensure_ascii=False)
        os.replace(tmp_path, self.ocr_cache_path)
        # Debug
        # print(f"[OCR-CACHE] Saved to {self.ocr_cache_path} ({len(self.ocr_cache)} entries)")

    def clear_cache(self) -> None:
        self.ocr_cache.clear()
        if os.path.exists(self.ocr_cache_path):
            try:
                os.remove(self.ocr_cache_path)
                print(f"[OCR-CACHE] Cleared {self.ocr_cache_path}")
            except OSError:
                pass

    @staticmethod
    def _hash_array(arr: np.ndarray) -> str:
        h = hashlib.sha1()
        h.update(arr.tobytes())
        return h.hexdigest()

    def _get_cached_ocr(self, key: str, content_hash: str) -> Optional[Dict[str, Any]]:
        entry = self.ocr_cache.get(key)
        if entry and entry.get('hash') == content_hash:
            return entry.get('ocr')
        return None

    def _set_cached_ocr(self, key: str, content_hash: str, ocr_payload: Dict[str, Any]) -> None:
        self.ocr_cache[key] = {'hash': content_hash, 'ocr': ocr_payload}

    # ---------------- Geometry ----------------
    def filter_overlapping_blocks(self, blocks: List[Any]) -> List[Any]:
        filtered_blocks: List[Any] = []
        for block in blocks:
            keep = True
            x1 = int(block.block.x_1); y1 = int(block.block.y_1)
            x2 = int(block.block.x_2); y2 = int(block.block.y_2)
            area1 = max(0, x2 - x1) * max(0, y2 - y1)

            to_remove: List[int] = []
            for i, other in enumerate(filtered_blocks):
                ox1 = int(other.block.x_1); oy1 = int(other.block.y_1)
                ox2 = int(other.block.x_2); oy2 = int(other.block.y_2)
                area2 = max(0, ox2 - ox1) * max(0, oy2 - oy1)

                ix1 = max(x1, ox1); iy1 = max(y1, oy1)
                ix2 = min(x2, ox2); iy2 = min(y2, oy2)
                iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                intersection_area = iw * ih

                if intersection_area > 0:
                    min_area = max(1, min(area1, area2))
                    overlap_ratio = intersection_area / min_area
                    if overlap_ratio > self.overlap_threshold:
                        if area1 > area2:
                            to_remove.append(i)
                        else:
                            keep = False
                            break

            if keep:
                for i in reversed(to_remove):
                    del filtered_blocks[i]
                filtered_blocks.append(block)

        return filtered_blocks

    # ---------------- Segmentation ----------------
    def segment_block(self, block, page_index: int, block_id: int, image_np: np.ndarray):
        block_type = str(block.type).lower()
        offset = int(self.offsets.get(block_type, 0))

        x1 = max(int(block.block.x_1) - offset, 0)
        y1 = max(int(block.block.y_1) - offset, 0)
        x2 = min(int(block.block.x_2) + offset, image_np.shape[1])
        y2 = min(int(block.block.y_2) + offset, image_np.shape[0])

        segment = image_np[y1:y2, x1:x2]
        if segment.size == 0:
            segment = np.ones((32, 32, 3), dtype=np.uint8) * 255

        h, w, c = segment.shape
        canvas = np.ones((h + 2 * self.padding, w + 2 * self.padding, c), dtype=np.uint8) * 255
        canvas[self.padding:self.padding + h, self.padding:self.padding + w] = segment
        segment = canvas

        seg_name = f"page{page_index}_block{block_id}_{block.type}"
        seg_img_path = os.path.join(self.segment_block_dir, f"{seg_name}.png")
        text_path = os.path.join(self.text_output_dir, f"{seg_name}.txt")
        txt_path = os.path.join(self.txt_folder, f"{seg_name}.txt")
        img_output_path = os.path.join(self.img_folder, f"{seg_name}.png")
        text_raw_path = os.path.join(self.text_raw_folder, f"{seg_name}.txt")

        cv2.imwrite(seg_img_path, cv2.cvtColor(segment, cv2.COLOR_RGB2BGR))
        return seg_name, seg_img_path, text_path, txt_path, img_output_path, text_raw_path, segment, block_type

    # ---------------- OCR ----------------
    def _run_or_load_ocr(self, seg_name: str, segment_rgb: np.ndarray, seg_img_path: str) -> Dict[str, Any]:
        """Run OCR on ndarray or load from cache."""
        content_hash = self._hash_array(segment_rgb)
        cached = self._get_cached_ocr(seg_name, content_hash)
        if cached is not None:
            return cached
        try:
            # IMPORTANT: pass ndarray, not path
            lines_with_scores = self.ocr_engine.extract_text_with_confidence(segment_rgb)
        except Exception as e:
            lines_with_scores = [{'line_text': f"[OCR Failed: {e}]", 'avg_confidence': 0.0, 'words': []}]
        payload = {
            'lines': lines_with_scores,
            'joined_text': '\n'.join([x.get('line_text', '') for x in lines_with_scores if str(x.get('line_text', '')).strip()]),
            'stats': {
                'num_lines': len(lines_with_scores),
                'avg_conf_over_lines': float(np.mean([x.get('avg_confidence', 0.0) for x in lines_with_scores]) if lines_with_scores else 0.0),
            },
        }
        self._set_cached_ocr(seg_name, content_hash, payload)
        return payload

    def get_ocr_payload(self, seg_name: str, segment_rgb: np.ndarray, seg_img_path: str) -> Dict[str, Any]:
        """Public wrapper for consumers to obtain cached OCR payload."""
        return self._run_or_load_ocr(seg_name, segment_rgb, seg_img_path)

    # ---------------- Markdown ----------------
    def generate_markdown(self, seg_name, seg_img_path, text_path, txt_path, img_output_path, text_raw_path, segment, block_type, page_index, block_id, first_title_used, ocr_payload):
        lines_with_scores = ocr_payload.get('lines', [])
        if getattr(self.ocr_engine, 'add_score', False):
            texts = [f"{entry.get('line_text','')} (conf: {entry.get('avg_confidence', 0.0):.2f})" for entry in lines_with_scores if str(entry.get('line_text', '')).strip()]
        else:
            texts = [entry.get('line_text', '') for entry in lines_with_scores if str(entry.get('line_text', '')).strip()]
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(texts))
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(texts))
        markdown_block = ""
        updated_first_title_used = first_title_used
        if block_type == 'title':
            header_prefix = '#' if not first_title_used else '##'
            markdown_block = f"{header_prefix} {' '.join(texts)}\n"
            updated_first_title_used = True
            with open(text_raw_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(texts))
        elif block_type == 'figure':
            cv2.imwrite(img_output_path, cv2.cvtColor(segment, cv2.COLOR_RGB2BGR))
            rel_img_path = os.path.relpath(img_output_path, self.segment_output_dir)
            markdown_block = f"![{seg_name}]({rel_img_path})\n"
        else:
            with open(text_raw_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(texts))
            markdown_block = f"{' '.join(texts)}\n"
        ocr_json_path = os.path.join(self.text_raw_folder, f"{seg_name}.ocr.json")
        try:
            with open(ocr_json_path, 'w', encoding='utf-8') as jf:
                json.dump(ocr_payload, jf, ensure_ascii=False)
        except Exception:
            pass
        return markdown_block, updated_first_title_used

    def process_block(self, block, page_index, block_id, image_np, first_title_used):
        seg_name, seg_img_path, text_path, txt_path, img_output_path, text_raw_path, segment, block_type = self.segment_block(block, page_index, block_id, image_np)
        ocr_payload = self._run_or_load_ocr(seg_name, segment, seg_img_path)
        markdown_block, updated_flag = self.generate_markdown(seg_name, seg_img_path, text_path, txt_path, img_output_path, text_raw_path, segment, block_type, page_index, block_id, first_title_used, ocr_payload)
        self._flush_ocr_cache()
        return markdown_block, updated_flag, ocr_payload
