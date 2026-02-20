#!/usr/bin/env python3
"""Run SAM3 from HuggingFace on test.png and save overlay + per-instance masks."""

import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image as PILImage
from transformers import Sam3Model, Sam3Processor

REPO_ROOT   = Path(__file__).parent.parent
TEXT_PROMPT = "road"
DOWNSCALE   = 4
THRESHOLD   = 0.3   # query score threshold
MASK_THRESH = 0.5   # pixel binarization threshold

print("Loading SAM3 from HuggingFace ...")
model     = Sam3Model.from_pretrained("facebook/sam3").to('cuda').eval()
processor = Sam3Processor.from_pretrained("facebook/sam3")

img_bgr = cv2.imread(str(REPO_ROOT / "test.png"))
h0, w0  = img_bgr.shape[:2]
img_bgr = cv2.resize(img_bgr, (w0 // DOWNSCALE, h0 // DOWNSCALE), interpolation=cv2.INTER_AREA)
h, w    = img_bgr.shape[:2]
print(f"Image: {w0}x{h0} -> {w}x{h}")

pil_img = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
inp     = processor(images=pil_img, text=TEXT_PROMPT, return_tensors="pt").to('cuda')

print(f"Running HF model (prompt='{TEXT_PROMPT}') ...")
with torch.no_grad():
    out = model(**inp)

print(f"pred_masks:      {out.pred_masks.shape}")
print(f"pred_logits:     {out.pred_logits.shape}")
print(f"presence_logits: {out.presence_logits.shape}")

results = processor.post_process_instance_segmentation(
    out,
    threshold=THRESHOLD,
    mask_threshold=MASK_THRESH,
    target_sizes=[(h, w)],
)[0]

scores = results["scores"]
masks  = results["masks"]   # (N, h, w) bool
boxes  = results["boxes"]

print(f"\nInstances found: {len(scores)}")
for i, (s, b) in enumerate(zip(scores, boxes)):
    print(f"  [{i}] score={s:.4f}  box={b.int().tolist()}")

# Build colour overlay – each instance gets a distinct colour
overlay = img_bgr.copy().astype(np.float32)
colours = [
    [0, 200, 0], [0, 0, 220], [220, 0, 0],
    [0, 200, 200], [200, 0, 200], [200, 200, 0],
]
combined_mask = np.zeros((h, w), dtype=np.uint8)
for i, mask in enumerate(masks):
    m   = mask.cpu().numpy().astype(bool)
    col = np.array(colours[i % len(colours)], dtype=np.float32)
    overlay[m] = overlay[m] * 0.45 + col * 0.55
    combined_mask[m] = 255

overlay = overlay.clip(0, 255).astype(np.uint8)

cv2.imwrite(str(REPO_ROOT / "hf_result.png"),  overlay)
cv2.imwrite(str(REPO_ROOT / "hf_mask.png"),    combined_mask)
print(f"\nSaved: hf_result.png")
print(f"Saved: hf_mask.png")
