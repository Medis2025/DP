import os
import fitz  # PyMuPDF
import tempfile
from PIL import Image
from doctr.models import ocr_predictor
from doctr.io import DocumentFile


# === Input PDF Path ===
PDF_PATH = "/cluster/home/gw/MATRIX/regex_variants/segement/Matrix_reper/pdf_mat/test_data/Genetics_in_Medicine/Genetics in Medicine 13.pdf"


def convert_pdf_to_images(pdf_path, dpi=200):
    """Convert each PDF page to an image and return image paths."""
    doc = fitz.open(pdf_path)
    image_paths = []
    tmpdir = tempfile.mkdtemp()

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)
        image_path = os.path.join(tmpdir, f"page_{page_num + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)

    return image_paths


def run_doctr_on_images(image_paths):
    """Run Doctr OCR on a list of image paths and print results."""
    print("🔍 Loading OCR model...")
    model = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)

    # Load images into Doctr document format
    doc = DocumentFile.from_images(image_paths)
    results = model(doc)

    for i, (page, image_path) in enumerate(zip(results.pages, image_paths)):
        print(f"\n\n=== 📄 Page {i + 1}: {os.path.basename(image_path)} ===")
        pil_img = Image.open(image_path)
        width, height = pil_img.size

        exported = page.export()

        print("\n📝 Plain Text:")
        lines = []
        for block in exported["blocks"]:
            for line in block["lines"]:
                line_text = " ".join(word["value"] for word in line["words"])
                lines.append(line_text)
        print("\n".join(lines))

        print("\n🔍 Word-Level Details:")
        for block in exported["blocks"]:
            for line in block["lines"]:
                for word in line["words"]:
                    text = word["value"]
                    conf = word["confidence"]
                    (x_min, y_min), (x_max, y_max) = word["geometry"]

                    # Convert to absolute coordinates
                    abs_xmin = int(x_min * width)
                    abs_ymin = int(y_min * height)
                    abs_xmax = int(x_max * width)
                    abs_ymax = int(y_max * height)

                    print(f"Word: '{text}' | Conf: {conf:.2f} | "
                          f"Norm Box: ({x_min:.3f}, {y_min:.3f}) → ({x_max:.3f}, {y_max:.3f}) | "
                          f"Abs Box: ({abs_xmin}, {abs_ymin}) → ({abs_xmax}, {abs_ymax})")


if __name__ == "__main__":
    image_paths = convert_pdf_to_images(PDF_PATH)
    run_doctr_on_images(image_paths)
