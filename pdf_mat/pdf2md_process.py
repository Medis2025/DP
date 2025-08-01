import os
import layoutparser as lp
from PDF2MD.pdf2md import PDFToMarkdown

class LayoutMarkdownExporter:
    def __init__(self, base_dir, output_dir, layout_model=None, batchprocess=False):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.batchprocess = batchprocess

        self.layout_model = layout_model or lp.Detectron2LayoutModel(
            config_path="/cluster/home/gw/detectron2/config.yml",
            model_path="/cluster/home/gw/detectron2/layout_weights.pth",
            label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"},
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
            device="cuda"
        )

    def run(self):
        processor = PDFToMarkdown(
            pdf_path=self.base_dir,
            output_base_dir=self.output_dir,
            layout_model=self.layout_model,
            batch_process=self.batchprocess
        )

        if self.batchprocess:
            processor.process_all_pdfs()
        else:
            processor.process()


if __name__ == "__main__":
    base_dir = "/cluster/home/gw/data"
    output_dir = "/cluster/home/gw/MATRIX/regex_variants/segement/Matrix_reper/outputs/md_output"

    exporter = LayoutMarkdownExporter(
        base_dir=base_dir,
        output_dir=output_dir,
        layout_model=None,
        batchprocess=True
    )
    exporter.run()
