from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import numpy as np
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

class DoctrOCR:
    def __init__(
        self,
        det_arch: str = "db_resnet50",
        reco_arch: str = "crnn_vgg16_bn",
        conf_threshold: float = 0.8,
        add_score: bool = True,
        return_boxes: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.det_arch = det_arch
        self.reco_arch = reco_arch
        self.conf_threshold = float(conf_threshold)
        self.add_score = bool(add_score)
        self.return_boxes = bool(return_boxes)
        self.device = device
        self.model = None

    def load_model(self) -> None:
        if self.model is None:
            self.model = ocr_predictor(det_arch=self.det_arch, reco_arch=self.reco_arch, pretrained=True)
            if self.device and hasattr(self.model, "to"):
                try:
                    self.model.to(self.device)
                except Exception:
                    pass

    @staticmethod
    def _ensure_rgb_uint8_np(img: Any) -> Tuple[np.ndarray, Image.Image]:
        if isinstance(img, str):
            if not os.path.isfile(img):
                raise ValueError(f"Path does not exist: {img}")
            pil = Image.open(img).convert("RGB")
            arr = np.array(pil, dtype=np.uint8)
            return arr, pil
        if isinstance(img, Image.Image):
            pil = img.convert("RGB")
            arr = np.array(pil, dtype=np.uint8)
            return arr, pil
        if isinstance(img, np.ndarray):
            arr = img
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            arr = arr.astype(np.uint8)
            pil = Image.fromarray(arr, mode="RGB")
            return arr, pil
        raise ValueError(f"Unsupported image input type for OCR: {type(img)}")

    def extract_text_with_confidence(self, image_input: Any) -> List[Dict[str, Any]]:
        """
        Run OCR on a single image (ndarray/PIL/path). Returns list of line dicts.
        Includes robust fallbacks for different doctr versions of DocumentFile.
        """
        self.load_model()

        try:
            np_rgb, pil_img = self._ensure_rgb_uint8_np(image_input)

            # Robust construction of DocumentFile across doctr versions
            tmp_path = None
            try:
                # Preferred: pass numpy array
                doc = DocumentFile.from_images([np_rgb])
            except Exception as e_np:
                try:
                    # Fallback 1: pass PIL Image
                    doc = DocumentFile.from_images([pil_img])
                except Exception as e_pil:
                    # Fallback 2: write to a temp PNG and pass path
                    import tempfile
                    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    try:
                        pil_img.save(tmp, format="PNG")
                        tmp.flush(); tmp.close()
                        tmp_path = tmp.name
                        doc = DocumentFile.from_images(tmp_path)
                    except Exception as e_path:
                        # Aggregate the underlying errors for easier debugging
                        raise RuntimeError(
                            f"DocumentFile.from_images failed; numpy: {e_np}; PIL: {e_pil}; path: {e_path}"
                        )

            try:
                result = self.model(doc)
            finally:
                # Clean up temp file if we created one
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

            # handle doctr versions gently
            try:
                exported = result.pages[0].export()
            except Exception:
                exported = result.export().pages[0] if hasattr(result, "export") else result.pages[0].export()

            width, height = pil_img.size
            out_lines: List[Dict[str, Any]] = []

            blocks = exported.get("blocks", []) if isinstance(exported, dict) else getattr(exported, "blocks", [])
            for block in blocks:
                for line in block.get("lines", []):
                    words = line.get("words", [])
                    if not words:
                        continue

                    word_payload: List[Dict[str, Any]] = []
                    confs: List[float] = []
                    for w in words:
                        text = w.get("value", "")
                        conf = float(w.get("confidence", 0.0) or 0.0)
                        confs.append(conf)
                        geom = w.get("geometry")
                        if isinstance(geom, (list, tuple)) and len(geom) == 2:
                            (x_min, y_min), (x_max, y_max) = geom
                            x1 = int(float(x_min) * width)
                            y1 = int(float(y_min) * height)
                            x2 = int(float(x_max) * width)
                            y2 = int(float(y_max) * height)
                            bbox = [x1, y1, x2, y2]
                        else:
                            bbox = [0, 0, 0, 0]
                        if self.return_boxes:
                            word_payload.append({"text": text, "confidence": conf, "bbox": bbox})

                    avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
                    if avg_conf < self.conf_threshold:
                        continue

                    line_text = " ".join([w.get("text", "") for w in word_payload]) if word_payload else " ".join([w.get("value", "") for w in words])
                    entry: Dict[str, Any] = {
                        "line_text": line_text.strip(),
                        "avg_confidence": avg_conf,
                    }
                    if self.return_boxes:
                        entry["words"] = word_payload

                    # Optional line bbox if provided
                    lgeom = line.get("geometry")
                    if isinstance(lgeom, (list, tuple)) and len(lgeom) == 2:
                        (lx_min, ly_min), (lx_max, ly_max) = lgeom
                        lx1 = int(float(lx_min) * width)
                        ly1 = int(float(ly_min) * height)
                        lx2 = int(float(lx_max) * width)
                        ly2 = int(float(ly_max) * height)
                        entry["bbox"] = [lx1, ly1, lx2, ly2]

                    # Legacy modes
                    if not self.return_boxes and self.add_score:
                        entry = {"line_text": f"{entry['line_text']} (conf: {avg_conf:.2f})", "avg_confidence": avg_conf}
                    elif not self.return_boxes and not self.add_score:
                        entry = {"line_text": entry["line_text"]}

                    out_lines.append(entry)

            return out_lines

        except Exception as e:
            raise RuntimeError(f"Doctr OCR failure: {e}")
