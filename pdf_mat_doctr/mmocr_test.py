import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use CUDA:1

import fitz  # PyMuPDF
import cv2
import numpy as np
import layoutparser as lp
from tqdm import tqdm
from mmocr.apis import MMOCRInferencer
from sege.block_sorter import BlockSorter

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

        for folder in [self.segment_output_dir, self.text_output_dir, self.txt_folder, self.img_folder, self.text_raw_folder]:
            os.makedirs(folder, exist_ok=True)

        label_map = {0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"}
        self.model = lp.Detectron2LayoutModel(
            config_path="/cluster/home/gw/detectron2/config.yml",
            model_path="/cluster/home/gw/detectron2/layout_weights.pth",
            label_map=label_map,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
            device="cuda"
        )

        self.mmocr = MMOCRInferencer(det='DBNet', rec='SAR')

    def sharpen_image(self, image):
        kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    def upscale_segment(self, segment, target_size=(256, 256)):
        h, w = segment.shape[:2]
        scale = min(target_size[0] / h, target_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # 更换插值方式为不模糊的
        resized = cv2.resize(segment, target_size, interpolation=cv2.INTER_NEAREST)

        # 加上锐化
        resized = self.sharpen_image(resized)

        canvas = np.ones((target_size[0], target_size[1], 3), dtype=np.uint8) * 255
        top = (target_size[0] - new_h) // 2
        left = (target_size[1] - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = resized
        return canvas


    def process(self):
        doc = fitz.open(self.pdf_path)
        all_markdown_lines = []
        first_title_used = False

        for page_index in tqdm(range(len(doc)), desc="Processing pages"):
            pix = doc[page_index].get_pixmap(dpi=450)
            image_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            sorter = BlockSorter(image_np.shape[1])
            sorted_blocks = sorter.sort_blocks(self.model.detect(image_np))

            for block_id, block in enumerate(sorted_blocks):
                block_type = block.type.lower()
                offset = 10 if block_type == "title" else 5 if block_type in ["table", "figure"] else 3 if block_type == "text" else 0

                x1 = max(int(block.block.x_1) - offset, 0)
                y1 = max(int(block.block.y_1) - offset, 0)
                x2 = min(int(block.block.x_2) + offset, image_np.shape[1])
                y2 = min(int(block.block.y_2) + offset, image_np.shape[0])
                segment = image_np[y1:y2, x1:x2]

                if segment.size == 0:
                    segment = np.ones((32, 32, 3), dtype=np.uint8) * 255

                padding = 20
                h, w, c = segment.shape
                canvas = np.ones((h + 2 * padding, w + 2 * padding, c), dtype=np.uint8) * 255
                canvas[padding:padding + h, padding:padding + w] = segment
                segment = canvas

                segment = self.upscale_segment(segment, target_size=(256, 256))

                seg_name = f"page{page_index}_block{block_id}_{block.type}"
                seg_img_path = os.path.join(self.segment_output_dir, f"{seg_name}.png")
                text_path = os.path.join(self.text_output_dir, f"{seg_name}.txt")
                txt_path = os.path.join(self.txt_folder, f"{seg_name}.txt")
                img_output_path = os.path.join(self.img_folder, f"{seg_name}.png")
                text_raw_path = os.path.join(self.text_raw_folder, f"{seg_name}.txt")

                cv2.imwrite(seg_img_path, cv2.cvtColor(segment, cv2.COLOR_RGB2BGR))

                try:
                    result = self.mmocr(seg_img_path)
                    predictions = result['predictions'][0]
                    texts = predictions.get('rec_texts', [])
                    texts = [t for t in texts if t.strip()]
                except Exception as e:
                    print(f"\u26a0\ufe0f OCR failed on {seg_name}: {e}")
                    texts = [f"[OCR Failed: {e}]"]

                with open(text_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(texts))
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(texts))

                markdown_block = ""
                if block_type == "title":
                    header_prefix = "#" if not first_title_used else "##"
                    markdown_block = f"{header_prefix} {' '.join(texts)}\n"
                    first_title_used = True
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

                all_markdown_lines.append(markdown_block)

        with open(self.markdown_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(all_markdown_lines))

        print(f"\u2705 All segment images saved to: {self.segment_output_dir}")
        print(f"\u2705 All OCR texts saved to: {self.text_output_dir}")
        print(f"\u2705 Sorted text files saved to: {self.txt_folder}")
        print(f"\u2705 Images saved to: {self.img_folder}")
        print(f"\u2705 Text raw files saved to: {self.text_raw_folder}")
        print(f"\u2705 Markdown file saved to: {self.markdown_path}")


if __name__ == "__main__":
    pdf_path = "/cluster/home/gw/MATRIX/regex_variants/segement/Matrix_reper/offset_out_Imat/pdf_img/277_2014_Article_2090/277_2014_Article_2090.pdf"
    output_base_dir = "/cluster/home/gw/MATRIX/regex_variants/segement/Matrix_reper/test_output"
    converter = PDFToMarkdown(pdf_path, output_base_dir)
    converter.process()
