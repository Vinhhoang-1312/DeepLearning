import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from torchvision import transforms
from PIL import Image
from models_lib import CNNClassifier, SwinClassifier

CLS_BASE = "results/classification"
DEVICE = "cpu"
CLASSES = ["Rounded RBCs", "Ovalocytes", "Fragmented", "2 Overlapping", "3 Overlapping", "Burr Cells", "Teardrops", "Angled Cells", "Borderline Oval"]

def test_models():
    cnn = CNNClassifier(n_classes=9).to(DEVICE).eval()
    if os.path.exists(os.path.join(CLS_BASE, "CNN_Cls_best.pt")):
        cnn.load_state_dict(torch.load(os.path.join(CLS_BASE, "CNN_Cls_best.pt"), map_location=DEVICE))
        print("Loaded CNN")
    else:
        print("CNN weights not found")

    swin = SwinClassifier(n_classes=9).to(DEVICE).eval()
    try:
        if os.path.exists(os.path.join(CLS_BASE, "Swin_Cls_best.pt")):
            swin.load_state_dict(torch.load(os.path.join(CLS_BASE, "Swin_Cls_best.pt"), map_location=DEVICE))
            print("Loaded Swin")
    except Exception as e:
        print("Swin load error:", e)

    tf_cls = transforms.Compose([
        transforms.Resize((80, 80)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    test_dir = "test_classification"
    for img_name in os.listdir(test_dir)[:5]:
        img_path = os.path.join(test_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        tensor = tf_cls(img).unsqueeze(0)
        
        with torch.no_grad():
            out_cnn = cnn(tensor)
            _, pred_cnn = torch.max(out_cnn, 1)
            
            try:
                out_swin = swin(tensor)
                _, pred_swin = torch.max(out_swin, 1)
                swin_res = CLASSES[pred_swin.item()]
            except:
                swin_res = "ERROR"

        print(f"File: {img_name} -> CNN: {CLASSES[pred_cnn.item()]} | Swin: {swin_res}")

test_models()
