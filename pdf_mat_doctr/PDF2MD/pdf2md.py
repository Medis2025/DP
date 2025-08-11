import os
import fitz
import cv2
import numpy as np
import layoutparser as lp
from tqdm import tqdm
import sys

sys.path.append("/cluster/home/gw/MATRIX/regex_variants/segement/Matrix_reper/pdf_mat/sege")
from seg_processer import SegmentProcessor

class PDFToMarkdown:
    def __init__(self, pdf_path, output_base_dir, layout_model=None, processor=None, create_dirs=True, batch_process=False):
        self.pdf_path = pdf_path
        self.batch_process = batch_process
        self.output_base_dir = output_base_dir

        self.model = layout_model or lp.Detectron2LayoutModel(
            config_path="/cluster/home/gw/detectron2/config.yml",
            model_path="/cluster/home/gw/detectron2/layout_weights.pth",
            label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"},
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
            device="cuda"
        )

        self.processor = processor or SegmentProcessor()
        self.ocr = self.processor.ocr

        if not self.batch_process:
            self._init_paths(os.path.splitext(os.path.basename(pdf_path))[0])
            if create_dirs:
                self._create_dirs()

    def _init_paths(self, pdf_name):
        self.pdf_name = pdf_name
        self.segment_output_dir = os.path.join(self.output_base_dir, pdf_name)
        self.text_output_dir = os.path.join(self.segment_output_dir, "text")
        self.txt_folder = os.path.join(self.segment_output_dir, "txt")
        self.img_folder = os.path.join(self.segment_output_dir, "sege", "img")
        self.segment_img_output_dir = os.path.join(self.segment_output_dir, "sege_img")
        self.text_raw_folder = os.path.join(self.segment_output_dir, "sege", "text_raw")
        self.markdown_path = os.path.join(self.segment_output_dir, f"{pdf_name}.md")

    def _create_dirs(self):
        folders = [
            self.segment_img_output_dir,
            self.segment_output_dir,
            self.text_output_dir,
            self.txt_folder,
            self.img_folder,
            self.text_raw_folder
        ]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

    def get_all_pdf_paths(self, pdf_path):
        if self.batch_process:
            pdf_files = []
            for root, _, files in os.walk(pdf_path):
                for fname in files:
                    if fname.endswith(".pdf"):
                        pdf_files.append(os.path.join(root, fname))
            return pdf_files
        else:
            return [pdf_path]

    def sort_blocks_by_position(self, layout):
        return sorted(layout, key=lambda b: (b.block.y_1, b.block.x_1))

    def format_markdown_block(self, block_type, texts, segment, seg_name, text_raw_path, img_output_path):
        markdown_block = ""
        if block_type == "title":
            header_prefix = "#" if not self.first_title_used else "##"
            markdown_block = f"{header_prefix} {' '.join(texts)}\n"
            self.first_title_used = True
            with open(text_raw_path, "w", encoding="utf-8") as f:
                f.write("\n".join(texts))
        elif block_type == "figure":
            cv2.imwrite(img_output_path, cv2.cvtColor(segment, cv2.COLOR_RGB2BGR))
            rel_img_path = os.path.relpath(img_output_path, self.segment_output_dir)
            markdown_block = f"![{seg_name}]({rel_img_path})\n"
        else:
            with open(text_raw_path, "w", encoding="utf-8") as f:
                f.write("\n".join(texts))
            markdown_block = f"{' '.join(texts)}\n"

        return markdown_block

    def process(self):
        doc = fitz.open(self.pdf_path)
        all_markdown_lines = []
        self.first_title_used = False

        for page_index in tqdm(range(len(doc)), desc="Processing pages"):
            pix = doc[page_index].get_pixmap(dpi=150)
            image_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            layout = self.model.detect(image_np)
            sorted_blocks = self.sort_blocks_by_position(layout)

            for block_id, block in enumerate(sorted_blocks):
                block_type = block.type.lower()

                segment, x1, y1 = self.processor.apply_offset_and_padding(block, image_np)
                if segment is None:
                    continue

                seg_name = f"page{page_index}_block{block_id}_{block.type}"
                seg_img_path = os.path.join(self.segment_img_output_dir, f"{seg_name}.png")
                text_path = os.path.join(self.text_output_dir, f"{seg_name}.txt")
                txt_path = os.path.join(self.txt_folder, f"{seg_name}.txt")
                img_output_path = os.path.join(self.img_folder, f"{seg_name}.png")
                text_raw_path = os.path.join(self.text_raw_folder, f"{seg_name}.txt")

                cv2.imwrite(seg_img_path, cv2.cvtColor(segment, cv2.COLOR_RGB2BGR))

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

                    markdown_block = self.format_markdown_block(
                        block_type, texts, segment, seg_name, text_raw_path, img_output_path
                    )
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

    def process_all_pdfs(self):
        pdf_paths = self.get_all_pdf_paths(self.pdf_path)
        shared_processor = self.processor
        for pdf_file_path in tqdm(pdf_paths, desc="Batch processing PDFs"):
            try:
                pdf_name = os.path.splitext(os.path.basename(pdf_file_path))[0]
                print(f"\n📄 Processing: {pdf_file_path}")
                pdf2md = PDFToMarkdown(
                    pdf_file_path,
                    self.output_base_dir,
                    layout_model=self.model,
                    processor=shared_processor,
                    create_dirs=True,
                    batch_process=False
                )
                pdf2md.process()
            except Exception as e:
                print(f"❌ Failed to process {pdf_file_path}: {e}")

if __name__ == "__main__":
    input_path = "/path/to/pdf_or_folder"
    output_dir = "/path/to/output"

    batch = os.path.isdir(input_path)
    shared_processor = SegmentProcessor()

    pdf2md = PDFToMarkdown(
        pdf_path=input_path,
        output_base_dir=output_dir,
        processor=shared_processor,
        batch_process=batch
    )

    if batch:
        pdf2md.process_all_pdfs()
    else:
        pdf2md.process()
