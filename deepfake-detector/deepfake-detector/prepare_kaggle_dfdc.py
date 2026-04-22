import os
import zipfile
import json
import shutil
import sys
import subprocess

def prepare_kaggle_data():
    zip_path = 'train_sample_videos.zip'
    extract_path = 'kaggle_sample_extracted'
    real_dir = 'dataset/real'
    fake_dir = 'dataset/fake'

    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found.")
        print("Please download 'train_sample_videos.zip' from:")
        print("https://www.kaggle.com/c/deepfake-detection-challenge/data")
        print("and place it in this directory, then run this script again.")
        sys.exit(1)

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Find the metadata.json file
    metadata_path = None
    for root, dirs, files in os.walk(extract_path):
        if 'metadata.json' in files:
            metadata_path = os.path.join(root, 'metadata.json')
            break

    if not metadata_path:
        print("Error: metadata.json not found in the extracted files.")
        sys.exit(1)

    print("Loading metadata...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Ensure directories exist
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    print("Sorting videos into dataset/real and dataset/fake...")
    real_count = 0
    fake_count = 0

    base_extract_dir = os.path.dirname(metadata_path)

    for filename, info in metadata.items():
        src_path = os.path.join(base_extract_dir, filename)
        if not os.path.exists(src_path):
            continue

        label = info.get('label')
        if label == 'REAL':
            dst_path = os.path.join(real_dir, filename)
            shutil.copy2(src_path, dst_path)
            real_count += 1
        elif label == 'FAKE':
            dst_path = os.path.join(fake_dir, filename)
            shutil.copy2(src_path, dst_path)
            fake_count += 1

    print(f"Successfully sorted {real_count} REAL videos and {fake_count} FAKE videos.")
    print("Cleaning up extracted folder...")
    shutil.rmtree(extract_path)

    print("Data preparation complete! You can now train the model.")
    
    # Optionally trigger training
    print("\nStarting model training...")
    subprocess.run([sys.executable, "train_model.py"])

if __name__ == '__main__':
    prepare_kaggle_data()
