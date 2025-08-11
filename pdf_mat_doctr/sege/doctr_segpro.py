import os
import cv2
import numpy as np
import yaml

class BlockProcessor:
    def __init__(self, segment_output_dir, text_output_dir, txt_folder, img_folder, text_raw_folder, ocr_engine, config_path):
        self.segment_output_dir = segment_output_dir
        self.text_output_dir = text_output_dir
        self.txt_folder = txt_folder
        self.img_folder = img_folder
        self.text_raw_folder = text_raw_folder
        self.ocr_engine = ocr_engine

        # Load configuration from YAML
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.padding = config.get('padding', 200)
        self.offsets = config.get('offsets', {
            'title': 10,
            'table': 5,
            'figure': 5,
            'text': 3
        })

    def filter_overlapping_blocks(self, blocks):
        filtered_blocks = []
        for block in blocks:
            keep = True
            x1 = int(block.block.x_1)
            y1 = int(block.block.y_1)
            x2 = int(block.block.x_2)
            y2 = int(block.block.y_2)
            area1 = (x2 - x1) * (y2 - y1)

            to_remove = []
            for i, other in enumerate(filtered_blocks):
                ox1 = int(other.block.x_1)
                oy1 = int(other.block.y_1)
                ox2 = int(other.block.x_2)
                oy2 = int(other.block.y_2)
                area2 = (ox2 - ox1) * (oy2 - oy1)

                # Check for overlap
                if not (x2 <= ox1 or x1 >= ox2 or y2 <= oy1 or y1 >= oy2):
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

    def segment_block(self, block, page_index, block_id, image_np):
        block_type = block.type.lower()
        offset = self.offsets.get(block_type, 0)

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
        seg_img_path = os.path.join(self.segment_output_dir, f"{seg_name}.png")
        text_path = os.path.join(self.text_output_dir, f"{seg_name}.txt")
        txt_path = os.path.join(self.txt_folder, f"{seg_name}.txt")
        img_output_path = os.path.join(self.img_folder, f"{seg_name}.png")
        text_raw_path = os.path.join(self.text_raw_folder, f"{seg_name}.txt")

        cv2.imwrite(seg_img_path, cv2.cvtColor(segment, cv2.COLOR_RGB2BGR))

        return seg_name, seg_img_path, text_path, txt_path, img_output_path, text_raw_path, segment, block_type

    def generate_markdown(self, seg_name, seg_img_path, text_path, txt_path, img_output_path, text_raw_path, segment, block_type, page_index, block_id, first_title_used):
        try:
            lines_with_scores = self.ocr_engine.extract_text_with_confidence(seg_img_path)
            if self.ocr_engine.add_score:
                texts = [f"{text} (conf: {conf:.2f})" for text, conf in lines_with_scores if text.strip()]
            else:
                texts = [text for text in lines_with_scores if text.strip()]
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

    def process_block(self, args):
        block, page_index, block_id, image_np, first_title_used = args

        # Filtering one block alone doesn't eliminate overlap — moved to outer logic
        outputs = self.segment_block(block, page_index, block_id, image_np)
        markdown_block = self.generate_markdown(*outputs, page_index, block_id, first_title_used=first_title_used)
        return markdown_block
