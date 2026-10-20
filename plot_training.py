import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_progress():
    csv_paths = [
        "runs/classify/rail_track_fault_cls/results.csv",
        "runs/classify/rail_track_fault_cls-2/results.csv",
        "runs/classify/rail_cls_fixed/results.csv"
    ]
    csv_path = None
    for p in csv_paths:
        if os.path.exists(p):
            csv_path = p
            break

    if not csv_path:
        print("Results CSV not found!")
        return
    
    df = pd.read_csv(csv_path)
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    
    # Ensure static/assets exists
    os.makedirs("static/assets", exist_ok=True)
    
    # 1. Accuracy vs Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['metrics/accuracy_top1'], marker='o', linestyle='-', color='#2ecc71', label='Accuracy')
    plt.title('Model Accuracy over Epochs', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('static/assets/accuracy_curve.png')
    plt.close()
    
    # 2. Loss vs Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train/loss'], marker='o', linestyle='-', color='#e74c3c', label='Train Loss')
    plt.plot(df['epoch'], df['val/loss'], marker='o', linestyle='-', color='#3498db', label='Val Loss')
    plt.title('Training and Validation Loss', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('static/assets/loss_curve.png')
    plt.close()

    print("Training plots saved to static/assets/")

if __name__ == "__main__":
    plot_training_progress()
