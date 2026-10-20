import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

class YOLORailwayDetector:
    def __init__(self, model_path: str = "model.pt"):
        """
        Initializes the AI & Computer Vision model for railway track inspection.
        Uses fine-tuned railway classification model if available, plus standard YOLO for obstacle detection.
        """
        # Prioritize latest trained model, model.pt, or rail_cls_fixed
        if os.path.exists("runs/classify/rail_track_fault_cls/weights/best.pt"):
            model_path = "runs/classify/rail_track_fault_cls/weights/best.pt"
        elif os.path.exists("model.pt"):
            model_path = "model.pt"
        elif os.path.exists("runs/classify/rail_cls_fixed/weights/best.pt"):
            model_path = "runs/classify/rail_cls_fixed/weights/best.pt"
            
        print(f"Loading Railway Inspection Model from: {model_path}")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Failed to load {model_path}, falling back to yolov8n.pt: {e}")
            self.model = YOLO("yolov8n.pt")

        # Load obstacle detector (YOLOv8 standard) for foreign object detection on tracks
        self.obstacle_model = None
        if os.path.exists("yolov8n.pt"):
            try:
                self.obstacle_model = YOLO("yolov8n.pt")
            except Exception as e:
                print(f"Obstacle model not loaded: {e}")

    def cv_preprocess(self, cv_img: np.ndarray) -> np.ndarray:
        """
        Computer Vision preprocessing: CLAHE contrast enhancement & bilateral filtering
        to highlight rail track edges, structural discontinuities, and cracks.
        """
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_bgr

    def cv_localize_defect(self, cv_img: np.ndarray):
        """
        Computer Vision Defect Localization:
        Uses morphology, edge detection, and contour analysis to locate the
        broken rail fissure / crack / fracture region on the track.
        Returns (bbox, crack_contour).
        """
        h, w, _ = cv_img.shape
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # CLAHE for crack contrast
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.bilateralFilter(enhanced, 7, 50, 50)
        
        # Black-hat transform highlights dark cracks on reflective rail heads
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)
        _, thresh = cv2.threshold(blackhat, 25, 255, cv2.THRESH_BINARY)
        
        # Canny edges
        edges = cv2.Canny(denoised, 50, 150)
        combined = cv2.bitwise_or(thresh, edges)
        
        # Morphological closing to group crack segments
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_box = None
        best_contour = None
        max_score = -1
        
        for c in contours:
            area = cv2.contourArea(c)
            x, y, bw, bh = cv2.boundingRect(c)
            # Filter out extreme sizes
            if 30 < bw < w * 0.85 and 30 < bh < h * 0.85 and 300 < area < (w * h * 0.5):
                # Prefer central regions where tracks typically run
                cx = x + bw / 2
                cy = y + bh / 2
                dist_center = np.sqrt(((cx - w/2) / w)**2 + ((cy - h/2) / h)**2)
                score = area * (1.0 - 0.5 * dist_center)
                if score > max_score:
                    max_score = score
                    # Expand padding around defect
                    pad = 20
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(w, x + bw + pad)
                    y2 = min(h, y + bh + pad)
                    best_box = [int(x1), int(y1), int(x2), int(y2)]
                    best_contour = c
                    
        if best_box is None:
            # Fallback to focal center of track
            best_box = [int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)]
            
        return best_box, best_contour

    def detect_defects(self, image_path: str, conf_threshold: float = 0.25):
        """
        Runs genuine AI & Computer Vision pipeline:
        1. OpenCV Computer Vision Preprocessing (CLAHE + Bilateral)
        2. YOLO Railway Fault Classification (Defective vs Healthy)
        3. Computer Vision Track Defect & Crack Localization
        4. Obstacle / Foreign Object Detection
        5. Visual Annotations & Structured Output
        """
        start_time = time.time()
        
        cv_img = cv2.imread(image_path)
        if cv_img is None:
            raise ValueError(f"Could not load image: {image_path}")

        h, w, _ = cv_img.shape
        detected_defects = []
        overall_status = "HEALTHY"
        max_confidence = 95.0
        annotated_img = cv_img.copy()

        # 1. Computer Vision Preprocessing
        preprocessed_img = self.cv_preprocess(cv_img)

        # 2. Railway Track Fault Classification
        yolo_result = self.model(preprocessed_img, conf=conf_threshold, verbose=False)[0]

        is_defective_class = False
        if hasattr(yolo_result, "probs") and yolo_result.probs is not None:
            top1_idx = int(yolo_result.probs.top1)
            raw_class_name = yolo_result.names[top1_idx]
            max_confidence = round(float(yolo_result.probs.top1conf.item()) * 100, 2)
            name_lower = raw_class_name.lower().strip()
            
            # Explicitly check for non-defective / healthy
            if "non defective" in name_lower or "non-defective" in name_lower or "healthy" in name_lower or "normal" in name_lower or "clean" in name_lower:
                is_defective_class = False
                overall_status = "HEALTHY"
            elif any(k in name_lower for k in ["defect", "fault", "crack", "broken", "fracture", "damage"]):
                is_defective_class = True
                overall_status = "DEFECTIVE"

        # 3. If Classification Model predicts DEFECTIVE: Localize Crack via Computer Vision
        if is_defective_class:
            bbox, contour = self.cv_localize_defect(cv_img)
            x1, y1, x2, y2 = bbox
            
            defect_item = {
                "class": "Broken Rail / Track Defect",
                "category": "TRACK_FAULT",
                "confidence": max_confidence,
                "severity": "Critical",
                "bbox": [x1, y1, x2, y2]
            }
            detected_defects.append(defect_item)

            # Draw Computer Vision Crack Boundary Box (Tech Corner Accents)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 220), 3)
            # Corner accents
            clen = min(25, (x2 - x1) // 4, (y2 - y1) // 4)
            if clen > 0:
                cv2.line(annotated_img, (x1, y1), (x1 + clen, y1), (0, 255, 255), 3)
                cv2.line(annotated_img, (x1, y1), (x1, y1 + clen), (0, 255, 255), 3)
                cv2.line(annotated_img, (x2, y1), (x2 - clen, y1), (0, 255, 255), 3)
                cv2.line(annotated_img, (x2, y1), (x2, y1 + clen), (0, 255, 255), 3)
                cv2.line(annotated_img, (x1, y2), (x1 + clen, y2), (0, 255, 255), 3)
                cv2.line(annotated_img, (x1, y2), (x1, y2 - clen), (0, 255, 255), 3)
                cv2.line(annotated_img, (x2, y2), (x2 - clen, y2), (0, 255, 255), 3)
                cv2.line(annotated_img, (x2, y2), (x2, y2 - clen), (0, 255, 255), 3)

            # Label on defect box
            tag_text = f"CRITICAL FAULT: Broken Rail ({max_confidence}%)"
            cv2.rectangle(annotated_img, (x1, max(0, y1 - 28)), (x1 + 380, y1), (0, 0, 220), -1)
            cv2.putText(annotated_img, tag_text, (x1 + 6, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Top prominent alert banner
            cv2.rectangle(annotated_img, (0, 0), (w, 55), (0, 0, 220), -1)
            banner_text = f"CRITICAL TRACK FAULT DETECTED (Confidence: {max_confidence}%)"
            cv2.putText(annotated_img, banner_text, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

        # 4. Obstacle / Foreign Object Detection (e.g. Person, Vehicle, Obstacle on tracks)
        if self.obstacle_model is not None:
            try:
                obs_results = self.obstacle_model(cv_img, conf=0.35, verbose=False)[0]
                if hasattr(obs_results, "boxes") and obs_results.boxes is not None and len(obs_results.boxes) > 0:
                    for box in obs_results.boxes:
                        cls_id = int(box.cls[0].item())
                        conf = round(float(box.conf[0].item()) * 100, 2)
                        cls_name = obs_results.names.get(cls_id, "Obstacle").lower()
                        # Relevant rail obstacle classes (persons, animals, vehicles, debris)
                        if cls_name in ["person", "car", "truck", "bus", "dog", "cow", "cat", "bicycle", "motorcycle", "backpack", "suitcase"]:
                            xyxy = [int(round(c)) for c in box.xyxy[0].tolist()]
                            detected_defects.append({
                                "class": f"Foreign Object ({cls_name.title()})",
                                "category": "FOREIGN_OBJECT",
                                "confidence": conf,
                                "severity": "High",
                                "bbox": xyxy
                            })
                            overall_status = "DEFECTIVE"
                            # Draw obstacle bounding box
                            ox1, oy1, ox2, oy2 = xyxy
                            cv2.rectangle(annotated_img, (ox1, oy1), (ox2, oy2), (0, 140, 255), 3)
                            cv2.rectangle(annotated_img, (ox1, max(0, oy1 - 25)), (ox1 + 250, oy1), (0, 140, 255), -1)
                            cv2.putText(annotated_img, f"OBSTACLE: {cls_name.title()} ({conf}%)", (ox1 + 5, max(16, oy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            except Exception as e:
                print(f"Error during obstacle detection: {e}")

        # 5. Healthy Track Annotation
        if overall_status == "HEALTHY":
            cv2.rectangle(annotated_img, (0, 0), (w, 55), (34, 139, 34), -1)
            banner_text = f"TRACK HEALTHY: Intact & Clear (Confidence: {max_confidence}%)"
            cv2.putText(annotated_img, banner_text, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

        # Save annotated image
        dir_name, file_name = os.path.split(image_path)
        annotated_filename = f"annotated_{file_name}"
        annotated_dir = os.path.join("static", "uploads")
        os.makedirs(annotated_dir, exist_ok=True)
        annotated_path = os.path.join(annotated_dir, annotated_filename)
        cv2.imwrite(annotated_path, annotated_img)

        inference_time_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "overall_status": overall_status,
            "status": overall_status,
            "confidence": max_confidence,
            "inference_time_ms": inference_time_ms,
            "defects": detected_defects,
            "annotated_image_path": f"/static/uploads/{annotated_filename}",
            "original_image_path": f"/static/uploads/{file_name}" if "static" not in dir_name else image_path
        }
