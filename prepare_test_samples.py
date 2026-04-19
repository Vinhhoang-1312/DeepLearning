import os, shutil, random
from pathlib import Path

# --- CONFIG ---
DATASET_BASE = r"c:\Users\DELL\Desktop\Vinh Hoang\Master Program\Học sâu\Project\Dataset"
OUT_DIR = r"c:\Users\DELL\Desktop\Vinh Hoang\Master Program\Học sâu\Project\test_samples"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"--- Preparing Test Samples in {OUT_DIR} ---")

# 1. Segmentation Samples (Images that models haven't seen)
seg_img_dir = os.path.join(DATASET_BASE, "RBC_Segmentation_Dataset-master", "train", "images")
if os.path.exists(seg_img_dir):
    imgs = sorted([os.path.join(seg_img_dir, f) for f in os.listdir(seg_img_dir) if f.endswith('.jpg')])
    # We used 80% for train, so the last 20% is test/val
    test_slice = imgs[int(0.8 * len(imgs)):]
    chosen = random.sample(test_slice, min(5, len(test_slice)))
    for f in chosen:
        shutil.copy(f, os.path.join(OUT_DIR, f"seg_test_{os.path.basename(f)}"))
    print(f"✅ Copied {len(chosen)} segmentation test images.")

# 2. Classification Samples (Optional, though app starts with full smears)
cls_dir = os.path.join(DATASET_BASE, "Elsafty_RBCs_for_Classification", "Cropped images")
if os.path.exists(cls_dir):
    classes = [d for d in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, d))]
    for c in classes:
        sample_path = os.path.join(cls_dir, c)
        imgs = sorted([os.path.join(sample_path, f) for f in os.listdir(sample_path) if f.endswith('.png')])
        # We used 70% train, 15% val, 15% test. So the last 15% is test.
        test_slice = imgs[int(0.85 * len(imgs)):]
        if test_slice:
            f = random.choice(test_slice)
            shutil.copy(f, os.path.join(OUT_DIR, f"cls_test_{c}_{os.path.basename(f)}"))
    print(f"✅ Copied sample cropped images from each class test set.")

print("\n🚀 DONE! You can now upload these images to the Streamlit app to test.")
