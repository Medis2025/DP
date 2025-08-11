# doctr_ocr.py
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

class DoctrOCR:
    def __init__(self, det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', conf_threshold=0.8, add_score=True):
        self.det_arch = det_arch
        self.reco_arch = reco_arch
        self.conf_threshold = conf_threshold
        self.add_score = add_score
        self.model = None  # Lazy loading for multiprocessing safety

    def load_model(self):
        if self.model is None:
            self.model = ocr_predictor(det_arch=self.det_arch, reco_arch=self.reco_arch, pretrained=True)

    def extract_text_with_confidence(self, image_path):
        self.load_model()
        try:
            doc = DocumentFile.from_images(image_path)
            result = self.model(doc)
            blocks = result.pages[0].blocks
            lines_with_scores = []
            lines_without_scores = []
            for block in blocks:
                for line in block.lines:
                    line_text = " ".join([word.value for word in line.words])
                    confidences = [word.confidence for word in line.words if word.confidence is not None]
                    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                    if avg_conf >= self.conf_threshold:
                        lines_with_scores.append((line_text, avg_conf))
                        lines_without_scores.append((line_text))
            return lines_with_scores if self.add_score else lines_without_scores
        except Exception as e:
            raise RuntimeError(f"Doctr OCR failure: {e}")
