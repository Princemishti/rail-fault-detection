import cv2
import numpy as np
from PIL import Image
import os

# Colors for Severities (BGR format for OpenCV)
SEVERITY_COLORS_BGR = {
    "Critical": (0, 0, 220),     # Bright Red
    "High": (0, 140, 255),       # Orange
    "Medium": (0, 215, 255),     # Yellow
    "Low": (50, 205, 50)         # Green
}

SEVERITY_COLORS_HEX = {
    "Critical": "#dc3545",
    "High": "#fd7e14",
    "Medium": "#ffc107",
    "Low": "#28a745"
}

def preprocess_image(image_path: str, denoise: bool = True, enhance_contrast: bool = True) -> np.ndarray:
    """
    Applies computer vision preprocessing: noise reduction, contrast enhancement (CLAHE).
    Returns processed BGR image matrix.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    # 1. Noise Reduction
    if denoise:
        # Fast Non-Local Means Denoising for color images
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # 2. Contrast Enhancement (CLAHE on L-channel of LAB color space)
    if enhance_contrast:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return img

def draw_defect_annotations(img: np.ndarray, defects: list) -> np.ndarray:
    """
    Draws professional industrial bounding boxes, class labels, confidence scores, 
    and severity indicators on the image.
    """
    annotated = img.copy()
    h, w, _ = annotated.shape

    for idx, defect in enumerate(defects):
        bbox = defect["bbox"]  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Clamp coordinates within image dimensions
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        severity = defect.get("severity", "Medium")
        color = SEVERITY_COLORS_BGR.get(severity, (0, 255, 255))
        label_text = f"{defect['class']} {defect['confidence']:.1f}% ({severity})"

        # Draw main bounding box with rounded corners / thick border
        line_thickness = max(2, int(min(w, h) / 300))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

        # Draw box corner accents for high-tech industrial aesthetic
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        if corner_len > 0:
            # Top-Left
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, line_thickness + 2)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, line_thickness + 2)
            # Top-Right
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, line_thickness + 2)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, line_thickness + 2)
            # Bottom-Left
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, line_thickness + 2)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, line_thickness + 2)
            # Bottom-Right
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, line_thickness + 2)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, line_thickness + 2)

        # Background tag rectangle for label text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(w, h) / 1000.0)
        font_thickness = max(1, int(font_scale * 2))
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

        tag_y1 = max(0, y1 - text_height - 10)
        tag_y2 = y1 if y1 - text_height - 10 >= 0 else y1 + text_height + 10
        cv2.rectangle(annotated, (x1, tag_y1), (x1 + text_width + 12, tag_y2), color, -1)

        # White or black contrast text
        text_color = (255, 255, 255) if severity in ["Critical", "High"] else (0, 0, 0)
        text_pos_y = tag_y2 - baseline - 3 if y1 - text_height - 10 >= 0 else tag_y2 - 5
        cv2.putText(annotated, label_text, (x1 + 6, text_pos_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return annotated
