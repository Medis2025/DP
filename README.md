# Layout-Aware PDF-to-Markdown Extraction Pipeline

This repository implements an efficient and modular pipeline for converting PDF documents into structured Markdown format. The system integrates layout detection using Detectron2 and text recognition via PaddleOCR, optimized for local processing with significantly improved performance.

Compared to large-scale solutions such as **MinerU**, this pipeline demonstrates a **200% speedup** on local benchmarks while retaining comparable structural fidelity and text accuracy.

---

## Overview

This system extracts structural information and textual content from scientific or technical PDF documents, reconstructing the content in human-readable and layout-preserving Markdown format.

### Main Components

1. **PDF Page Rendering**: Converts PDF pages to high-resolution images (150 DPI) using PyMuPDF.
2. **Layout Detection**: Applies a pre-trained Detectron2 model to segment each page into block types such as `title`, `text`, `list`, `table`, and `figure`.
3. **Block Segmentation**: Extracts cropped images of each block with configurable padding for improved OCR quality.
4. **Text Recognition**: Uses PaddleOCR to recognize text from each segmented block.
5. **Markdown Assembly**: Integrates recognized text and image references into a structured Markdown document.

---

## Features

- Batch processing of entire PDF folders
- High-accuracy layout detection and text recognition
- Shared model instances to minimize GPU memory usage
- Export of structured Markdown files along with intermediate results (text and images)
- Modular design for extension or integration into larger systems

---

## System Architecture

```
PDF → Pages → Detectron2 Layout → Block Cropping → PaddleOCR → Markdown Rendering
```

Each PDF is processed page-by-page with layout-aware segmentation followed by OCR, then reconstructed into Markdown with appropriate structural tags and image references.

---

## Setup

### Dependencies
````
documents\paddle_install.pdf
````
view this to properly install paddlepaddle and other dependencies, recommand to use ````python == 3.10 ````

### Repository Structure

```
PDF2MD/
├── pdf2md.py           # Core PDF-to-Markdown processor
├── seg_processer.py    # Segment processor with OCR logic
├── run.py              # Batch/CLI entry point
```

---

## Usage

### Batch Mode

```bash
python run.py --input /path/to/pdf_folder --output /path/to/output_dir
```

### Single File

```bash
python run.py --input /path/to/file.pdf --output /path/to/output_dir
```

### Programmatic API

```python
from PDF2MD.pdf2md import PDFToMarkdown
from seg_processer import SegmentProcessor
import layoutparser as lp

layout_model = lp.Detectron2LayoutModel(...)
processor = SegmentProcessor()

pdf2md = PDFToMarkdown(
    pdf_path="/path/to/file_or_folder",
    output_base_dir="/path/to/output",
    layout_model=layout_model,
    processor=processor,
    batch_process=True
)
pdf2md.process_all_pdfs()
```

---

## Output Format

Each PDF generates the following structure:

```
output/
└── example.pdf/
    ├── example.md              # Final Markdown document
    ├── text/                   # Clean OCR texts
    ├── txt/                    # Ordered OCR blocks
    ├── sege_img/               # Saved block images
    └── sege/
        ├── img/                # Cropped input images
        └── text_raw/           # Unformatted raw text
```

---

## Performance

On a typical academic corpus of biomedical PDFs:

- Compared to **MinerU**, this system achieved:
  - **~200% faster inference** on a single GPU (A6000)
  - Lower peak memory usage due to shared model architecture
  - Comparable output fidelity in structure and text

Optimizations include:

- Shared layout and OCR model instances
- Selective memory cleanup after block-level processing
- Avoidance of redundant disk I/O or model reinitialization

---

## Recommendations

- Use DPI ≥ 150 for clear OCR results
- Avoid re-initializing PaddleOCR in loops; use shared instances
- For large batch processing, periodically release GPU memory using:
  
  ```python
  import gc
  import paddle
  gc.collect()
  paddle.device.cuda.empty_cache()
  ```

- Consider using `multiprocessing` for further isolation and memory control

---

## License

This project is released for academic and research purposes. Please cite or acknowledge the authors if used in downstream research or system integration.

````
@misc{medis2025layoutmd,
  author       = {Medis Lab},
  title        = {Layout-Aware PDF-to-Markdown Extraction Pipeline},
  year         = {2025},
  howpublished = {\url{https://github.com/Medis2025/DP}},
  note         = {Department of Biomedical Informatics, [MedicineAI Ltd.]}
}
````

---

## Contact

For questions or feedback, please contact the project maintainer.

Email: medis2025@outlook.com