import os
from typing import List

class PDFFinder:
    def __init__(self, pdf_base_dir: str, output_base_dir: str):
        self.pdf_base_dir = pdf_base_dir
        self.output_base_dir = output_base_dir

    def find_pdfs(self) -> List[str]:
        pdf_files = []
        for root, _, files in os.walk(self.pdf_base_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        return pdf_files

    def is_processed_pdf(self, pdf_path: str) -> bool:
        # Extract pdf name without extension
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.join(self.output_base_dir, pdf_name)
        return os.path.exists(output_dir)

    def get_unprocessed_pdfs(self) -> List[str]:
        all_pdfs = self.find_pdfs()
        return [pdf for pdf in all_pdfs if not self.is_processed_pdf(pdf)]

# Example usage:
# finder = PDFFinder(
#     pdf_base_dir="/cluster/home/gw/data",
#     output_base_dir="/cluster/home/gw/MATRIX/regex_variants/segement/Matrix_reper/test_output/pdf_img"
# )
# unprocessed_pdfs = finder.get_unprocessed_pdfs()
# for pdf in unprocessed_pdfs:
#     print(pdf)
