import os

# Fixed project structure (as per your requirement)
BASE_INPUT_PATH = r"data_analysis\datasets\audiofiles\CAT_AUDIO_Dataset"
BASE_OUTPUT_PATH = r"collected_data\cleaned_data\cleaned_audiodata"
OUTPUT_FILE = os.path.join(BASE_OUTPUT_PATH, "audio_dataset_full_report.txt")

# Supported audio formats
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a")

def analyze_audio_dataset(base_path):
    """
    Project-specific reporting script.
    Analyzes:
    - Total audio files
    - Total size in MB
    - Classes
    - Files per class
    - mp3 / wav / other counts
    """

    if not os.path.exists(base_path):
        print("Error: Input folder does not exist:", base_path)
        return None

    total_files = 0
    total_size_bytes = 0

    mp3_count = 0
    wav_count = 0
    other_audio_count = 0

    class_file_counts = {}
    all_classes = set()

    # Walk recursively
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(AUDIO_EXTENSIONS):
                total_files += 1
                file_path = os.path.join(root, file)
                total_size_bytes += os.path.getsize(file_path)

                # Format counts
                if file.lower().endswith(".mp3"):
                    mp3_count += 1
                elif file.lower().endswith(".wav"):
                    wav_count += 1
                else:
                    other_audio_count += 1

                # Class detection (top folder)
                relative_path = os.path.relpath(root, base_path)
                class_name = relative_path.split(os.sep)[0]
                all_classes.add(class_name)

                if class_name not in class_file_counts:
                    class_file_counts[class_name] = 0
                class_file_counts[class_name] += 1

    total_size_mb = total_size_bytes / (1024 * 1024)

    return {
        "total_files": total_files,
        "total_size_mb": total_size_mb,
        "mp3_count": mp3_count,
        "wav_count": wav_count,
        "other_audio_count": other_audio_count,
        "classes": sorted(all_classes),
        "class_file_counts": class_file_counts
    }

def save_report(stats):
    os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        f.write("AUDIO DATASET FULL ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write("Software Engineering View:\n")
        f.write("This is a project-specific reporting script.\n")
        f.write("Good for:\n")
        f.write("- One project\n")
        f.write("- One folder layout\n")
        f.write("- One-time report\n\n")

        f.write(f"Input Folder: {BASE_INPUT_PATH}\n\n")

        f.write("Overall Dataset Statistics:\n")
        f.write(f"Total Audio Files: {stats['total_files']}\n")
        f.write(f"Total Dataset Size: {stats['total_size_mb']:.2f} MB\n\n")

        f.write("Audio Format Distribution:\n")
        f.write(f"MP3 Files: {stats['mp3_count']}\n")
        f.write(f"WAV Files: {stats['wav_count']}\n")
        f.write(f"Other Audio Files: {stats['other_audio_count']}\n\n")

        f.write("Class Distribution Analysis:\n")
        f.write(f"Number of Classes: {len(stats['classes'])}\n")
        f.write("Classes:\n")

        for cls in stats["classes"]:
            count = stats["class_file_counts"].get(cls, 0)
            f.write(f"- {cls}: {count} files\n")

        f.write("\nWhy this matters:\n")
        f.write("Files per class is useful for:\n")
        f.write("- Class imbalance analysis\n")
        f.write("- Dataset bias checking\n\n")

        f.write("Conclusion:\n")
        f.write("This script provides a complete project-level audit of the audio dataset,\n")
        f.write("including size, formats, and class distribution for ML readiness.\n")

    return OUTPUT_FILE

if __name__ == "__main__":
    stats = analyze_audio_dataset(BASE_INPUT_PATH)

    if stats:
        report_path = save_report(stats)

        # Terminal summary
        print("\n===== AUDIO DATASET SUMMARY =====")
        print(f"Total Audio Files: {stats['total_files']}")
        print(f"Total Size: {stats['total_size_mb']:.2f} MB")
        print(f"MP3 Files: {stats['mp3_count']}")
        print(f"WAV Files: {stats['wav_count']}")
        print(f"Other Audio Files: {stats['other_audio_count']}")
        print(f"Number of Classes: {len(stats['classes'])}")
        print("Files per class:")
        for cls, count in stats["class_file_counts"].items():
            print(f"  {cls}: {count}")

        print("\nReport saved at:")
        print(report_path)
