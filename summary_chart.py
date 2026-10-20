import matplotlib.pyplot as plt
import os

def create_summary_chart():
    # Values from our evaluation
    metrics = ['Precision', 'Recall', 'F1-Score']
    values = [0.9091, 0.9091, 0.9091]
    
    os.makedirs("static/assets", exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, values, color=['#27ae60', '#2980b9', '#8e44ad'])
    plt.ylim(0, 1.1)
    plt.title('Final Model Performance (Defective Class)', fontsize=14)
    plt.ylabel('Score')
    
    # Add labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('static/assets/summary_metrics.png')
    plt.close()

if __name__ == "__main__":
    create_summary_chart()
