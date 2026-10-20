import os

def count_images():
    if os.path.exists("dataset"):
        print("Dataset images in INT-345-Project-main/dataset:")
        total = 0
        for split in sorted(os.listdir("dataset")):
            split_dir = os.path.join("dataset", split)
            if not os.path.isdir(split_dir):
                continue
            for cls in sorted(os.listdir(split_dir)):
                cls_dir = os.path.join(split_dir, cls)
                if os.path.isdir(cls_dir):
                    count = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    print(f"  {split} / {cls}: {count}")
                    total += count
        print(f"  Total Images: {total}")
        return

    with open("dataset_path.txt", "r") as f:
        dataset_path = f.read().strip()
    
    root = os.path.join(dataset_path, "Railway Track fault Detection Updated")
    splits = ["Train", "Test", "Validation"]
    classes = ["Defective", "Non defective"]
    
    for split in splits:
        split_dir = os.path.join(root, split)
        if not os.path.exists(split_dir):
            print(f"Split dir {split_dir} does not exist.")
            continue
        for cls in classes:
            class_dir = os.path.join(split_dir, cls)
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"{split} / {cls}: {count}")
            else:
                print(f"Class dir {class_dir} does not exist.")

if __name__ == "__main__":
    count_images()
