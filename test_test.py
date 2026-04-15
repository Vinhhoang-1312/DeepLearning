import os
import random
import zipfile
from pathlib import Path

DATASET_BASE = r"c:\Users\DELL\Desktop\Vinh Hoang\Master Program\Học sâu\Project\Dataset"
SLIDE_DIRS = [
    "Elsafty_RBCs_for_Segmentation_and_Detection_Slide_2",
    "Elsafty_RBCs_for_Segmentation_and_Detection_Slide_3",
]
MAX_SAMPLES = 5000

def collect_paths(base, slide_dirs, max_samples):
    imgs, masks = [], []
    for sd in slide_dirs:
        sd_path = Path(base) / sd
        
        # 1. Tự động kiểm tra file ảnh (nhanh, dùng next() thay vì load nguyên list gây quá tải RAM)
        has_imgs = next(sd_path.rglob("*.png"), None) is not None or next(sd_path.rglob("*.jpg"), None) is not None
        
        if not has_imgs:
            print(f"Tiến hành giải nén tự động cho: {sd} ...")
            zips = list(sd_path.glob("*.zip"))
            if not zips:
                print(f"⚠ Bỏ qua {sd_path}: Thư mục không có ảnh và không có file .zip!")
                continue
            for zip_path in zips:
                print(f"  Đang giải nén: {zip_path.name}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(sd_path)
                    
        # 2. Phân loại ảnh/mask nhanh gọn
        print(f"Đang duyệt ảnh trong thư mục: {sd} ...")
        img_files, mask_files = [], []
        allowed_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
        
        for p in sd_path.rglob("*.*"):
            if not p.is_file() or p.name.startswith("."): 
                continue
            if p.suffix.lower() not in allowed_exts:
                continue
                
            path_str = str(p.parent).lower()
            if "mask" in path_str or "label" in path_str:
                mask_files.append(p)
            elif "crop" in path_str or "image" in path_str:
                img_files.append(p)

        if not img_files or not mask_files:
            print(f"⚠ Bỏ qua {sd}: Không phân loại được ảnh/mask (Tìm thấy: {len(img_files)} crop, {len(mask_files)} mask).")
            continue
            
        # 3. Ghép cặp ảnh-mask
        mask_dict = {m.stem: str(m) for m in mask_files}
        matched = 0
        for img_p in img_files:
            stem = img_p.stem
            if stem in mask_dict:
                imgs.append(str(img_p))
                masks.append(mask_dict[stem])
                matched += 1
                
        if matched == 0:
            print(f"⚠ Cảnh báo {sd}: Tìm thấy {len(img_files)} crop và {len(mask_files)} mask nhưng không có cặp nào khớp tên!")
            
    if len(imgs) == 0:
        raise ValueError("Data rỗng! Không có cặp ảnh/mask nào được phân loại thành công.")
        
    idx = random.sample(range(len(imgs)), min(max_samples, len(imgs)))
    return [imgs[i] for i in idx], [masks[i] for i in idx]

img_paths, mask_paths = collect_paths(DATASET_BASE, SLIDE_DIRS, MAX_SAMPLES)
print(f"Total samples after pairing: {len(img_paths)}")
