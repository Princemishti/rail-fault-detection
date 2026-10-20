import matplotlib.pyplot as plt
import numpy as np
import os

def create_visualizations():
    # Metrics from our evaluation
    precision = 0.9091
    recall = 0.9091
    f1_score = 0.9091
    accuracy = 0.9091
    
    tp, fp, fn, tn = 10, 1, 1, 10
    
    # Ensure static/assets exists
    os.makedirs("static/assets", exist_ok=True)
    
    # 1. Bar Chart for Metrics
    plt.figure(figsize=(10, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1_score]
    colors = ['#3498db', '#2ecc71', '#f1c40f', '#e67e22']
    
    plt.bar(metrics, values, color=colors)
    plt.ylim(0, 1.1)
    plt.title('Model Performance Metrics', fontsize=16)
    plt.ylabel('Score')
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2%}", ha='center', fontweight='bold')
    plt.savefig('static/assets/metrics_bar.png')
    plt.close()
    
    # 2. Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    cm = np.array([[tp, fn], [fp, tn]])
    labels = ['Defective', 'Healthy']
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    
    # Add numbers to the heatmap
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'g'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=14)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('static/assets/confusion_matrix.png')
    plt.close()
    
    # 3. Pie Chart of Correct vs Incorrect
    plt.figure(figsize=(8, 8))
    correct = tp + tn
    incorrect = fp + fn
    plt.pie([correct, incorrect], labels=['Correct', 'Incorrect'], 
            autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=140, explode=(0, 0.1))
    plt.title('Prediction Accuracy Distribution')
    plt.savefig('static/assets/accuracy_pie.png')
    plt.close()

    print("Visualizations saved to static/assets/")

if __name__ == "__main__":
    create_visualizations()
