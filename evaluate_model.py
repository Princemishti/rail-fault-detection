from ultralytics import YOLO
import os
import json

def evaluate_model():
    # Load the best model
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
    # Note: imgsz must match what was used in training (224)
    print("Running validation on the test set...")
    metrics = model.val(data=data_dir, split='test', imgsz=224)
    
    # Extract key metrics
    # For classification, metrics.top1 is accuracy
    # We can also get precision and recall if we calculate from confusion matrix or if available
    
    results = {
        "Top-1 Accuracy": float(metrics.top1),
        "Top-5 Accuracy": float(metrics.top5),
        "Fitness": float(metrics.fitness),
        "Mean Results": metrics.results_dict
    }
    
    print("\n--- Model Performance Summary ---")
    print(f"Accuracy (Top-1): {results['Top-1 Accuracy']:.4f}")
    print(f"Accuracy (Top-5): {results['Top-5 Accuracy']:.4f}")
    
    # Detailed report
    print("\nDetailed Metrics:")
    for key, value in results['Mean Results'].items():
        print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    evaluate_model()
