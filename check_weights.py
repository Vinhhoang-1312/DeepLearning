import torch
import os

RESULTS_DIR = r"c:\Users\DELL\Desktop\Vinh Hoang\Master Program\Học sâu\Project\results"

def check_file(path):
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return
    print(f"\n--- Checking: {os.path.basename(path)} ---")
    state = torch.load(path, map_location='cpu')
    for k, v in state.items():
        print(f"{k:35} | Shape: {list(v.shape)}")

# Check CNN
cnn_path = os.path.join(RESULTS_DIR, "classification", "CNN_Cls_best.pt")
check_file(cnn_path)

# Check RNN
rnn_path = os.path.join(RESULTS_DIR, "classification", "RNN_Cls_best.pt")
check_file(rnn_path)

# Check ViT
vit_path = os.path.join(RESULTS_DIR, "classification", "ViT_Cls_best.pt")
check_file(vit_path)
