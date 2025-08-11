import os
import re
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict


class VariantExtractor:
    def __init__(
        self,
        variant_yml_path: str,
        ocr_engine,
        segment_processor,
        get_matrix_position_fn,
        save_root_dir: str = "matrix",
        block_color_map: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    ) -> None:
        self.variant_patterns = self._load_variant_patterns_from_yml(variant_yml_path)
        self.ocr_engine = ocr_engine
        self.segment_processor = segment_processor
        self.get_matrix_position = get_matrix_position_fn
        self.save_root_dir = save_root_dir
        self.block_color_map = block_color_map or {
            "text": (0.8, 0.9, 1.0, 0.3),
            "title": (1.0, 0.8, 0.8, 0.3),
            "table": (0.8, 1.0, 0.8, 0.3),
            "figure": (0.9, 0.9, 0.9, 0.3),
        }

    # --------------------
    # Config & patterns
    # --------------------
    def _load_variant_patterns_from_yml(self, path: str) -> List[re.Pattern]:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        patterns = config.get('variant_patterns_ignore_sep', [])
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    @staticmethod
    def _match_variants_in_text(text: str, patterns: List[re.Pattern]) -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                matches.append({
                    "variant": match.group(),
                    "start": int(match.start()),
                    "end": int(match.end()),
                    "context": text[max(0, match.start() - 50): match.end() + 50],
                    "pattern": pattern.pattern,
                })
        return matches

    # --------------------
    # Helpers
    # --------------------
    @staticmethod
    def _to_py_int(x: Any) -> int:
        return int(x) if not isinstance(x, int) else x

    @staticmethod
    def _stringify_matrix_keys(mat: Dict[Tuple[int, int], List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        return {f"{int(r)},{int(c)}": vals for (r, c), vals in mat.items()}

    def _get_ocr_payload(self, seg_name: str, segment_rgb: np.ndarray, seg_img_path: str) -> Dict[str, Any]:
        if hasattr(self.segment_processor, 'get_ocr_payload'):
            return self.segment_processor.get_ocr_payload(seg_name, segment_rgb, seg_img_path)
        if hasattr(self.segment_processor, '_run_or_load_ocr'):
            return self.segment_processor._run_or_load_ocr(seg_name, segment_rgb, seg_img_path)
        # Fallback: try engine directly
        try:
            lines_with_scores = self.ocr_engine.extract_text_with_confidence(segment_rgb)
        except Exception as e:
            lines_with_scores = [{
                'line_text': f"[OCR Failed: {e}]",
                'avg_confidence': 0.0,
                'words': [],
            }]
        return {
            'lines': lines_with_scores,
            'joined_text': '\n'.join([x.get('line_text', '') for x in lines_with_scores if str(x.get('line_text', '')).strip()]),
            'stats': {
                'num_lines': len(lines_with_scores),
                'avg_conf_over_lines': float(np.mean([x.get('avg_confidence', 0.0) for x in lines_with_scores]) if lines_with_scores else 0.0),
            },
        }

    # --------------------
    # Core
    # --------------------
    def extract_and_plot_variants(self, image_np: np.ndarray, layout, page_index: int, pdf_name: str, pdf_dir: str):
        """Returns (matrix_jsonable, variant_jsonable). Writes raw/overlay/text files.

        Fixes:
        - **No duplicate plotting**: one box per regex match (group words by match span).
        - **Accurate coordinates**: account for BlockProcessor's `offset` and `padding`.
        """
        matrix: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        variant_matrix: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)

        fig, ax = plt.subplots(figsize=(12, 16))
        ax.imshow(image_np)
        ax.axis('off')

        collected_lines: List[str] = []
        H, W = image_np.shape[:2]

        # Fetch transform params from the block processor
        pad = int(getattr(self.segment_processor, 'padding', 0))
        offsets_cfg = getattr(self.segment_processor, 'offsets', {}) or {}

        for block_id, block in enumerate(layout):
            label = str(block.type).lower()

            # Segment first (creates a padded canvas around the crop)
            seg_name, seg_img_path, text_path, txt_path, img_output_path, text_raw_path, segment, block_type = \
                self.segment_processor.segment_block(block, page_index, block_id, image_np)
            if segment is None:
                continue

            # Original (un-offset) block bounds on the page
            bx1 = max(int(block.block.x_1), 0)
            by1 = max(int(block.block.y_1), 0)
            bx2 = min(int(block.block.x_2), W)
            by2 = min(int(block.block.y_2), H)

            # The crop used for OCR included an expansion by `offset` and then a `padding` around it.
            offset_val = int(offsets_cfg.get(block_type, offsets_cfg.get(label, 0)))
            crop_x1 = max(bx1 - offset_val, 0)
            crop_y1 = max(by1 - offset_val, 0)
            crop_x2 = min(bx2 + offset_val, W)
            crop_y2 = min(by2 + offset_val, H)

            # For matrix cell placement, use the original block bounds
            row, col = self.get_matrix_position([bx1, by1, bx2, by2], image_np.shape)
            matrix[(row, col)].append({
                "type": label.upper(),
                "bbox": (self._to_py_int(bx1), self._to_py_int(by1), self._to_py_int(bx2), self._to_py_int(by2)),
                "seg_name": seg_name,
            })

            matched_block = False

            # OCR payload
            try:
                ocr_payload = self._get_ocr_payload(seg_name, segment, seg_img_path)
                lines = ocr_payload.get('lines', [])
            except Exception as e:
                print(f"⚠️ OCR payload unavailable for {seg_name}: {e}")
                lines = []

            # De-dup set: one emission per (seg,line_idx,start,end)
            seen_matches = set()

            for line_idx, line in enumerate(lines):
                line_text = str(line.get('line_text', '')).strip()
                if not line_text:
                    continue
                collected_lines.append(line_text)

                # Find regex matches once
                variant_matches = self._match_variants_in_text(line_text, self.variant_patterns)
                if not variant_matches:
                    continue

                # Build word tokens and their spans in the line_text **exactly as the OCR created it**
                words = line.get('words') or []
                tokens = []  # (text, bbox_canvas, start, end)
                cursor = 0
                for w in words:
                    t = w.get('text') if isinstance(w, dict) else None
                    if not t:
                        t = w.get('value', '') if isinstance(w, dict) else ''
                    t = str(t)
                    start = cursor
                    end = start + len(t)
                    bbox = w.get('bbox') if isinstance(w, dict) else None
                    tokens.append((t, bbox, start, end))
                    cursor = end + 1  # single space between words in OCR line_text

                # Helper to map a bbox from canvas (padded crop) to page coordinates
                def canvas_to_page(box: List[int]) -> List[int]:
                    x1p = self._to_py_int(crop_x1 + box[0] - pad)
                    y1p = self._to_py_int(crop_y1 + box[1] - pad)
                    x2p = self._to_py_int(crop_x1 + box[2] - pad)
                    y2p = self._to_py_int(crop_y1 + box[3] - pad)
                    # clamp
                    x1p = max(0, min(W, x1p)); x2p = max(0, min(W, x2p))
                    y1p = max(0, min(H, y1p)); y2p = max(0, min(H, y2p))
                    return [x1p, y1p, x2p, y2p]

                for m in variant_matches:
                    key = (seg_name, line_idx, m['start'], m['end'])
                    if key in seen_matches:
                        continue

                    # Collect word boxes whose spans overlap the match span
                    span_boxes: List[List[int]] = []
                    for (t, bbox, ws, we) in tokens:
                        if bbox is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                            continue
                        # Overlap test on character spans
                        if not (we <= m['start'] or ws >= m['end']):
                            # this word intersects the match span
                            span_boxes.append(list(bbox))

                    # Fallback to line-level bbox if no word overlaps (e.g., punctuation-only match)
                    if not span_boxes:
                        line_bbox = line.get('bbox')
                        if isinstance(line_bbox, (list, tuple)) and len(line_bbox) == 4:
                            span_boxes = [list(line_bbox)]

                    if not span_boxes:
                        continue

                    # Union of boxes in canvas space -> single box
                    x1u = min(b[0] for b in span_boxes)
                    y1u = min(b[1] for b in span_boxes)
                    x2u = max(b[2] for b in span_boxes)
                    y2u = max(b[3] for b in span_boxes)
                    page_box = canvas_to_page([x1u, y1u, x2u, y2u])

                    # Emit once per match
                    variant_matrix[(row, col)].append({
                        "variant": m["variant"],
                        "bbox": page_box,
                        "type": label.upper(),
                        "context": m["context"],
                        "pattern": m["pattern"],
                        "seg_name": seg_name,
                    })
                    ax.add_patch(plt.Rectangle(
                        (page_box[0], page_box[1]),
                        max(1, page_box[2] - page_box[0]),
                        max(1, page_box[3] - page_box[1]),
                        fill=True, facecolor=(1.0, 1.0, 0.0, 0.5), edgecolor='orange', linewidth=2,
                    ))
                    ax.text(page_box[2] + 5, page_box[1], m['variant'], fontsize=9, color='black',
                            bbox=dict(facecolor='yellow', alpha=0.8, boxstyle='round'))

                    seen_matches.add(key)
                    matched_block = True

            if matched_block and label in self.block_color_map:
                ax.add_patch(plt.Rectangle(
                    (bx1, by1), max(1, bx2 - bx1), max(1, by2 - by1),
                    fill=True, facecolor=self.block_color_map[label], edgecolor='blue', linewidth=1,
                ))

        # Ensure dirs
        for sub in ["raw", "overlay", "text", self.save_root_dir]:
            os.makedirs(os.path.join(pdf_dir, sub), exist_ok=True)

        # Save artifacts
        raw_img_path = os.path.join(pdf_dir, "raw", f"page{page_index}.png")
        plot_img_path = os.path.join(pdf_dir, "overlay", f"page{page_index}_plotted.png")
        text_path = os.path.join(pdf_dir, "text", f"{page_index:04d}.txt")

        cv2.imwrite(raw_img_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        fig.savefig(plot_img_path, dpi=200, bbox_inches='tight')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(collected_lines))
        plt.close(fig)

        matrix_jsonable = self._stringify_matrix_keys(matrix)
        variant_jsonable = self._stringify_matrix_keys(variant_matrix)

        return matrix_jsonable, variant_jsonable
