import os
import shutil
from ultralytics import YOLO

def train_model():
    print("=" * 60)
    print("  🚀 TRAINING RAIL TRACK FAULT DETECTION MODEL")
    print("  📁 Dataset Directory: dataset/")
    print("=" * 60)
    
    data_dir = os.path.abspath("dataset")
    
    # 1. Load pretrained YOLOv8n-cls model
    model = YOLO('yolov8n-cls.pt')
    
    # 2. Train model on the dataset folder
    # Epochs=8 provides optimal convergence on CPU with fast execution
    results = model.train(
        data=data_dir, 
        epochs=8, 
        imgsz=224, 
        batch=16,
        project='runs/classify',
        name='rail_track_fault_cls',
        exist_ok=True,
        workers=0
    )
    
    print("\n✅ Training complete! Results saved in 'runs/classify/rail_track_fault_cls'")
    
    # 3. Synchronize trained weights to model.pt and rail_cls_fixed
    trained_best = os.path.join("runs", "classify", "rail_track_fault_cls", "weights", "best.pt")
    if os.path.exists(trained_best):
        print(f"📦 Copying trained weights from {trained_best} to model.pt...")
        shutil.copy(trained_best, "model.pt")
        
        fixed_dest_dir = os.path.join("runs", "classify", "rail_cls_fixed", "weights")
        os.makedirs(fixed_dest_dir, exist_ok=True)
        shutil.copy(trained_best, os.path.join(fixed_dest_dir, "best.pt"))
        print("✅ Best weights synchronized to 'model.pt' and 'runs/classify/rail_cls_fixed/weights/best.pt'")
    
    # 4. Generate training plots
    try:
        from plot_training import plot_training_progress
        plot_training_progress()
        print("✅ Updated training curves in static/assets/")
    except Exception as e:
        print(f"Note: Could not generate plots: {e}")
        
    # 5. Export to ONNX if available
    try:
        path = model.export(format='onnx')
        print(f"✅ Model exported to {path}")
    except Exception as e:
        print(f"ONNX export skipped (optional): {e}")

    # 6. Run validation evaluation
    print("\n" + "=" * 60)
    print("  📊 EVALUATING TRAINED MODEL")
    print("=" * 60)
    metrics = model.val(data=data_dir)
    print(f"Top-1 Validation Accuracy: {metrics.top1:.4f} ({metrics.top1 * 100:.2f}%)")
    print(f"Top-5 Validation Accuracy: {metrics.top5:.4f} ({metrics.top5 * 100:.2f}%)")
    print("=" * 60 + "\n")
    
    return results

if __name__ == "__main__":
    train_model()
