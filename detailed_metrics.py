from ultralytics import YOLO
import os
import numpy as np

def calculate_precision_recall():
    model = YOLO("model.pt")
    
    if os.path.exists("dataset"):
        data_dir = os.path.abspath("dataset")
    elif os.path.exists("dataset_path.txt"):
        with open("dataset_path.txt", "r") as f:
            dataset_path = f.read().strip()
        data_dir = os.path.join(dataset_path, "Railway Track fault Detection Updated")
    else:
        data_dir = "dataset"
    
    # Run validation on the TEST set
    results = model.val(data=data_dir, split='test', imgsz=224)
    
    # The confusion matrix is available in results.confusion_matrix
    # For classification, it's a matrix where [true_class, predicted_class]
    cm = results.confusion_matrix.matrix
    
    # Classes are: 0: Defective, 1: Non defective (based on alphabetic order usually)
    # Let's verify labels
    names = model.names
    print(f"Class Names: {names}")
    
    # Defective is usually 0, Non defective is 1
    # CM structure for 2 classes:
    # [[TP, FN],
    #  [FP, TN]]
    # or similar. YOLOv8 Classification CM: rows are true, cols are predicted.
    
    # Let's calculate for 'Defective' (Class 0)
    tp = cm[0, 0]
    fp = cm[1, 0]
    fn = cm[0, 1]
    tn = cm[1, 1]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n--- Detailed Detection Parameters (Class: Defective) ---")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN): {tn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Overall Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f"Overall Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    calculate_precision_recall()
