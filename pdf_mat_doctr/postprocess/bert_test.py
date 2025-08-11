# ocr_autofix.py

import torch
import re
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from spellchecker import SpellChecker


# ==== 1. 初始化模型和拼写器 ====
device = 0 if torch.cuda.is_available() else -1
model_path = "/cluster/home/gw/MATRIX/models/distil-biobert"  # 本地 BioBERT 路径

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path).to(f"cuda:{device}" if device == 0 else "cpu")
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=device)

spell = SpellChecker()


# ==== 2. 自动插入 [MASK] 的函数 ====
def insert_masks(text):
    # 粘连规则
    text = re.sub(r'(\d)([A-Za-z])', r'\1 [MASK] \2', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 [MASK] \2', text)

    # 拼写规则
    words = text.split()
    masked = []
    for word in words:
        if "[MASK]" in word or word.lower() in spell or re.match(r"[\(\)\.,0-9]", word):
            masked.append(word)
        else:
            masked.append("[MASK]")
    return " ".join(masked)


# ==== 3. 逐个填充每个 [MASK] 并替换 ====
def fill_and_replace(masked_text):
    words = masked_text.split()
    output_text = masked_text

    while "[MASK]" in output_text:
        # 只处理当前第一个 MASK
        segments = output_text.split("[MASK]", 1)
        prefix = segments[0]
        suffix = segments[1]

        temp_input = prefix + "[MASK]" + suffix
        results = fill_mask(temp_input)

        if isinstance(results, list) and isinstance(results[0], dict):
            top_choice = results[0]['token_str'].strip()
        elif isinstance(results, list) and isinstance(results[0], list):
            top_choice = results[0][0]['token_str'].strip()
        else:
            raise ValueError("Unexpected format from fill_mask output")

        # 替换第一个 MASK
        output_text = prefix + top_choice + suffix

    return output_text


# ==== 4. 测试样本 ====
if __name__ == "__main__":
    ocr_text = "Together with the recentresults of Sodhaetal(2000), it now appearsthatonly1of3 CHK2 mutationsoriginallyreported by"

    print("\n📜 Original OCR Text:")
    print(ocr_text)

    masked_text = insert_masks(ocr_text)
    print("\n🩹 Masked Text:")
    print(masked_text)

    revised = fill_and_replace(masked_text)
    print("\n✅ Final Revised Sentence:")
    print(revised)
