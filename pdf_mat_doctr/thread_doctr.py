import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use CUDA:1
import sys
import math
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# --- IMPORTANT: keep heavy imports INSIDE worker to avoid forking issues ---

def run_one(pdf_path: str, output_base_dir: str) -> str:
    """Process a single PDF path. Returns the output directory.
    Heavy libs are imported here so each process initializes its own GPU context."""
    # Minimize CPU thread contention per process
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    # Honor pre-set CUDA device (e.g., CUDA_VISIBLE_DEVICES=1 from caller)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

    # Lazy imports in worker
    import fitz  # PyMuPDF
    import numpy as np
    import layoutparser as lp
    from sege.block_sorter import BlockSorter
    from sege.doctr_markdown import BlockProcessor
    from PIL import Image
    import yaml
    import json

    from matrix.MatrixUtils import MatrixUtils
    from ocr_model.doctr_box import DoctrOCR
    from matrix.variant_extractor import VariantExtractor

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
            self.config_path = \
                "/cluster/home/gw/MATRIX/regex_variants/segement/Matrix_reper/pdf_mat/sege/seg_config.yml"

            for d in [self.segment_output_dir, self.raw_dir, self.overlay_dir, self.text_dir,
                      self.matrix_dir, self.txt_folder, self.img_folder, self.text_raw_folder]:
                os.makedirs(d, exist_ok=True)

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

        @staticmethod
        def _pixmap_to_rgb_numpy(pix: fitz.Pixmap):
            if pix.n >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            arr = np.frombuffer(pix.samples, dtype=np.uint8)
            return arr.reshape(pix.h, pix.w, 3)

        def process(self):
            doc = fitz.open(self.pdf_path)
            all_markdown_lines = []
            first_title_used = False

            for page_index in range(len(doc)):
                page = doc[page_index]
                pix = page.get_pixmap(dpi=450)
                image_rgb = self._pixmap_to_rgb_numpy(pix)

                # raw page
                from PIL import Image
                Image.fromarray(image_rgb).save(os.path.join(self.raw_dir, f"page{page_index}.png"))

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
                        print(f"[Worker] Block err p{page_index} b{block_id}: {e}")

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
                    print(f"[Worker] Variant err page {page_index}: {e}")

            try:
                with open(self.markdown_path, "w", encoding="utf-8") as f:
                    f.write("\n\n".join(all_markdown_lines))
            except Exception as e:
                print(f"[Worker] Markdown save failed: {e}")

            return self.segment_output_dir

    worker = PDFToMarkdown(pdf_path, output_base_dir)
    return worker.process()


def init_worker():
    """Top-level initializer for worker processes (picklable).
    Ignores SIGINT so the main process handles Ctrl+C cleanly.
    """
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main(pdf_base_dir: str, output_base_dir: str, workers: int = 2):
    """Parallel batch runner. Use a small number of workers (e.g., 2-3) for a single GPU.
    """
    from pdf_find.pdf_finder import PDFFinder
    from tqdm import tqdm

    finder = PDFFinder(pdf_base_dir, os.path.join(output_base_dir, "pdf_img"))
    all_pdfs = finder.find_pdfs()
    unprocessed = finder.get_unprocessed_pdfs()

    print(f"Total PDFs in base dir: {len(all_pdfs)}")
    print(f"Unprocessed PDFs: {len(unprocessed)}")
    if not unprocessed:
        return

    os.makedirs(os.path.join(output_base_dir, "pdf_img"), exist_ok=True)

    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as ex:
        futures = {ex.submit(run_one, pdf, output_base_dir): pdf for pdf in unprocessed}
        try:
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Batch x{workers}"):
                pdf = futures[fut]
                try:
                    outdir = fut.result()
                    print(f"[DONE] {pdf} -> {outdir}")
                except Exception as e:
                    print(f"[FAIL] {pdf}: {e}")
        except KeyboardInterrupt:
            print("\n[MAIN] KeyboardInterrupt: cancelling pending futures...")
            for f in futures:
                f.cancel()
            ex.shutdown(wait=False, cancel_futures=True)
            raise


if __name__ == "__main__":
    pdf_base_dir = "/cluster/home/gw/data"
    output_base_dir = "/cluster/home/gw/MATRIX/regex_variants/segement/Matrix_Reper/test_output"

    WORKERS = int(os.environ.get("PDF_WORKERS", "2"))

    main(pdf_base_dir, output_base_dir, workers=WORKERS)


