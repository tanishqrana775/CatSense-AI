import os

BASE_INPUT_PATH = r"data_analysis\datasets\imagefiles\IMAGES_Dataset"
BASE_OUTPUT_PATH = r"data_analysis\cleaned_data\cleaned_imagedata"
OUTPUT_FILE = os.path.join(BASE_OUTPUT_PATH, "image_dataset_full_report.txt")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp")

def analyze_image_dataset(base_path):
    if not os.path.exists(base_path):
        print("Error: Input folder does not exist:", base_path)
        return None

    total_files = 0
    total_size_bytes = 0

    split_counts = {}
    split_class_counts = {}
    all_extensions = set()

    for split in os.listdir(base_path):
        split_path = os.path.join(base_path, split)
        if not os.path.isdir(split_path):
            continue

        split_counts[split] = 0
        split_class_counts[split] = {}

        for cls in os.listdir(split_path):
            cls_path = os.path.join(split_path, cls)
            if not os.path.isdir(cls_path):
                continue

            split_class_counts[split][cls] = 0

            for file in os.listdir(cls_path):
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    total_files += 1
                    split_counts[split] += 1
                    split_class_counts[split][cls] += 1

                    file_path = os.path.join(cls_path, file)
                    total_size_bytes += os.path.getsize(file_path)

                    ext = os.path.splitext(file)[1].lower()
                    all_extensions.add(ext)

    total_size_mb = total_size_bytes / (1024 * 1024)

    return {
        "total_files": total_files,
        "total_size_mb": total_size_mb,
        "split_counts": split_counts,
        "split_class_counts": split_class_counts,
        "extensions": sorted(all_extensions)
    }

def save_report(stats):
    os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("IMAGE DATASET HIERARCHICAL ANALYSIS REPORT\n")
        f.write("=" * 55 + "\n\n")

        f.write("Input Folder:\n")
        f.write(f"{BASE_INPUT_PATH}\n\n")

        f.write("Overall Dataset Statistics:\n")
        f.write(f"Total Image Files: {stats['total_files']}\n")
        f.write(f"Total Dataset Size: {stats['total_size_mb']:.2f} MB\n\n")

        f.write("Split Distribution:\n")
        for split, count in stats["split_counts"].items():
            f.write(f"- {split}: {count} images\n")

        f.write("\nClass Distribution Inside Each Split:\n")

        for split, classes in stats["split_class_counts"].items():
            f.write(f"\n{split}:\n")
            for cls, count in classes.items():
                f.write(f"  {cls}: {count} images\n")

        f.write("\nImage File Extensions (all):\n")
        f.write(", ".join(stats["extensions"]) + "\n\n")

        f.write("Why this matters:\n")
        f.write("- Shows real ML class distribution\n")
        f.write("- Detects class imbalance per split\n")
        f.write("- Verifies train/test/val integrity\n\n")

        f.write("Conclusion:\n")
        f.write("This report provides a hierarchical audit of the image dataset\n")
        f.write("including split-level and class-level distributions.\n")

    return OUTPUT_FILE

if __name__ == "__main__":
    print("Scanning:", os.path.abspath(BASE_INPUT_PATH))

    stats = analyze_image_dataset(BASE_INPUT_PATH)

    if stats:
        report_path = save_report(stats)

        print("\n===== IMAGE DATASET SUMMARY =====")
        print(f"Total Images: {stats['total_files']}")
        print(f"Total Size: {stats['total_size_mb']:.2f} MB")

        print("\nSplits:")
        for split, count in stats["split_counts"].items():
            print(f"  {split}: {count}")

        print("\nClasses per split:")
        for split, classes in stats["split_class_counts"].items():
            print(f"\n{split}:")
            for cls, count in classes.items():
                print(f"  {cls}: {count}")

        print("\nReport saved at:")
        print(os.path.abspath(report_path))
