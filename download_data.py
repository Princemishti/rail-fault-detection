import kagglehub
import os

def download_dataset():
    print("Downloading dataset...")
    path = kagglehub.dataset_download("salmaneunus/railway-track-fault-detection")
    print("Path to dataset files:", path)
    return path

if __name__ == "__main__":
    path = download_dataset()
    # Create a symlink or record the path for easier access later
    with open("dataset_path.txt", "w") as f:
        f.write(path)
