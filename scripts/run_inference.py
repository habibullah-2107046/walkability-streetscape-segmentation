# run_inference.py
import os, glob
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"

print(f"Using device: {DEVICE}")
processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(DEVICE).eval()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "preprocessed")
MASK_DIR  = os.path.join(BASE_DIR, "masks")
SEG_DIR   = os.path.join(BASE_DIR, "segmented")

# Cityscapes 19-class color palette (standard)
PALETTE = np.array([
    [128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],
    [153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],
    [70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],
    [0,60,100],[0,80,100],[0,0,230],[119,11,32]
], dtype=np.uint8)

def colorize(mask):
    return PALETTE[mask]

@torch.no_grad()
def segment_image(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    mask = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return mask

if __name__ == "__main__":
    os.makedirs(MASK_DIR, exist_ok=True)
    os.makedirs(SEG_DIR, exist_ok=True)

    files = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
    print(f"Found {len(files)} preprocessed images to segment.")

    for fpath in tqdm(files, desc="Segmenting"):
        base_id = os.path.splitext(os.path.basename(fpath))[0]
        mask = segment_image(fpath)
        np.save(os.path.join(MASK_DIR, f"{base_id}_mask.npy"), mask)
        Image.fromarray(colorize(mask)).save(os.path.join(SEG_DIR, f"{base_id}_seg.jpg"))

    print("Inference complete.")