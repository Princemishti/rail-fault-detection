import requests
import os
import time
import random

def get_simulation_images():
    """
    Collects all images from dataset/train and dataset/val (both Defective and Non defective).
    Guarantees every single image in both train and val is included in sensor simulation.
    """
    train_defective = []
    train_healthy = []
    val_defective = []
    val_healthy = []
    test_defective = []
    test_healthy = []

    if os.path.exists("dataset"):
        for sub in sorted(os.listdir("dataset")):
            sub_dir = os.path.join("dataset", sub)
            if not os.path.isdir(sub_dir):
                continue
            sub_lower = sub.lower()
            split_tag = "VAL" if "val" in sub_lower else ("TRAIN" if "train" in sub_lower else "TEST")
            for root, _, files in os.walk(sub_dir):
                for f in sorted(files):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_p = os.path.join(root, f)
                        parent = os.path.basename(os.path.dirname(full_p)).lower()
                        is_def = "defective" in parent and "non defective" not in parent
                        item = (full_p, split_tag, "Defective" if is_def else "Non defective", f)
                        if split_tag == "TRAIN":
                            (train_defective if is_def else train_healthy).append(item)
                        elif split_tag == "VAL":
                            (val_defective if is_def else val_healthy).append(item)
                        else:
                            (test_defective if is_def else test_healthy).append(item)

    # Interleave train, val, and test
    all_images = []
    max_train = max(len(train_healthy), len(train_defective))
    for i in range(max_train):
        if i < len(train_healthy):
            all_images.append(train_healthy[i])
        if i < len(train_defective):
            all_images.append(train_defective[i])
            
    max_val = max(len(val_healthy), len(val_defective))
    for i in range(max_val):
        if i < len(val_healthy):
            all_images.append(val_healthy[i])
        if i < len(val_defective):
            all_images.append(val_defective[i])

    max_test = max(len(test_healthy), len(test_defective))
    for i in range(max_test):
        if i < len(test_healthy):
            all_images.append(test_healthy[i])
        if i < len(test_defective):
            all_images.append(test_defective[i])

    # Fallback to dataset_path.txt if local dataset folder is empty
    if not all_images and os.path.exists("dataset_path.txt"):
        with open("dataset_path.txt", "r") as f:
            dataset_path = f.read().strip()
        for root, _, files in os.walk(dataset_path):
            for f in sorted(files):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append((os.path.join(root, f), "DATASET", "Track", f))

    return all_images


def simulate_sensor():
    test_images = get_simulation_images()
    
    if not test_images:
        print("No images found in dataset/train or dataset/val to simulate!")
        return

    print(f"Starting IoT Sensor Simulation with {len(test_images)} images from dataset (train & val)...")
    url = "http://localhost:8000/predict"
    
    battery = 100.0
    current_idx = 0
    
    while True:
        battery -= random.uniform(0.1, 0.4)
        if battery < 10.0:
            print("Low battery! Simulation stopping for 'recharge'...")
            time.sleep(3)
            battery = 100.0
            continue
            
        img_path, split, cat, filename = test_images[current_idx % len(test_images)]
        current_idx += 1
        
        display_name = f"[{split}] {filename}"
        print(f"[{((current_idx - 1) % len(test_images)) + 1}/{len(test_images)}] Sensor sending: {display_name} ({cat}) | Battery: {battery:.1f}%")
        
        try:
            with open(img_path, "rb") as img_file:
                files = {"file": (display_name, img_file, "image/jpeg")}
                response = requests.post(url, files=files, timeout=10)
                
            if response.status_code == 200:
                result = response.json()
                print(f"  -> Prediction: {result.get('status')} ({result.get('confidence')}%) | Speed: {result.get('inference_time_ms', 'N/A')}ms")
            else:
                print(f"  -> Server Error: HTTP {response.status_code}")
        except Exception as e:
            print(f"  -> Server unreachable: {e}")
            
        time.sleep(2)


if __name__ == "__main__":
    simulate_sensor()
