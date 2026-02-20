#!/usr/bin/env python3
"""Quick inference test using the exported SAM3 JIT models."""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from transformers import Sam3Processor

REPO_ROOT   = Path(__file__).parent.parent
MODELS_DIR  = REPO_ROOT / "cpp" / "models"
TEXT_PROMPT = "road"
DOWNSCALE   = 4
THRESHOLD   = 0.3   # query score threshold  (matches HF post-processor default)
MASK_THRESH = 0.5   # pixel binarization threshold

# Load processor just for tokenization
print("Loading processor ...")
proc = Sam3Processor.from_pretrained("facebook/sam3")

# Load exported JIT models
print("Loading JIT models ...")
img_enc  = torch.jit.load(str(MODELS_DIR / "image_encoder.pt"),  map_location="cpu").to("cuda")
txt_enc  = torch.jit.load(str(MODELS_DIR / "text_encoder.pt"),   map_location="cpu").to("cuda")
mask_dec = torch.jit.load(str(MODELS_DIR / "mask_decoder.pt"),   map_location="cpu").to("cuda")

# Load and downscale test image
img_bgr = cv2.imread(str(REPO_ROOT / "test.png"))
h0, w0  = img_bgr.shape[:2]
img_bgr = cv2.resize(img_bgr, (w0 // DOWNSCALE, h0 // DOWNSCALE), interpolation=cv2.INTER_AREA)
h, w    = img_bgr.shape[:2]
print(f"Image: {w0}x{h0} -> {w}x{h}")

# Pre-process image: BGR->RGB, resize to 1008x1008, /255, normalize
IMG_SIZE = 1008
img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_res   = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
img_f     = img_res.astype(np.float32) / 255.0
for c in range(3):
    img_f[:, :, c] = (img_f[:, :, c] - 0.5) / 0.5
pixel_values = torch.from_numpy(img_f.transpose(2, 0, 1)[np.newaxis]).to('cuda')  # (1,3,1008,1008)

# Tokenize text prompt
text_inp       = proc(text=TEXT_PROMPT, return_tensors="pt")
input_ids      = text_inp.input_ids.to('cuda')
attention_mask = text_inp.attention_mask.to('cuda')
print(f"Tokens: {input_ids.shape}, prompt='{TEXT_PROMPT}'")

# Run inference
print("Running image encoder ...")
with torch.no_grad():
    vision_feats = img_enc(pixel_values)   # 8-tuple of FPN tensors

print("Running text encoder ...")
with torch.no_grad():
    text_embeds = txt_enc(input_ids, attention_mask)   # (1, T, 256)

print("Running mask decoder ...")
original_sizes = torch.tensor([[h, w]], dtype=torch.int64).to('cuda')
with torch.no_grad():
    pred_masks, pred_logits, presence_logits = mask_dec(
        *vision_feats, text_embeds, attention_mask, original_sizes
    )

print(f"pred_masks:      {pred_masks.shape}")
print(f"pred_logits:     {pred_logits.shape}")
print(f"presence_logits: {presence_logits.shape}")

# Post-process: mirror HF processor.post_process_instance_segmentation
scores = pred_logits.sigmoid() * presence_logits.sigmoid()  # (1, Q)
scores = scores.squeeze(0)                                   # (Q,)
masks  = pred_masks.sigmoid().squeeze(0)                     # (Q, H_m, W_m)

keep   = scores > THRESHOLD
scores = scores[keep]
masks  = masks[keep]   # (N, H_m, W_m)

print(f"\nInstances found: {len(scores)}")
for i, s in enumerate(scores):
    print(f"  [{i}] score={s:.4f}")

# Resize each mask to image size with bilinear then binarize
if len(masks) > 0:
    masks_up = F.interpolate(
        masks.unsqueeze(0),   # (1, N, H_m, W_m)
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)              # (N, h, w)
    masks_bin = (masks_up > MASK_THRESH)
else:
    masks_bin = torch.zeros(0, h, w, dtype=torch.bool)

# Build colour overlay – each instance gets a distinct colour
overlay = img_bgr.copy().astype(np.float32)
colours = [
    [0, 200, 0], [0, 0, 220], [220, 0, 0],
    [0, 200, 200], [200, 0, 200], [200, 200, 0],
]
combined_mask = np.zeros((h, w), dtype=np.uint8)
for i, mask in enumerate(masks_bin):
    m   = mask.cpu().numpy().astype(bool)
    col = np.array(colours[i % len(colours)], dtype=np.float32)
    overlay[m] = overlay[m] * 0.45 + col * 0.55
    combined_mask[m] = 255

overlay = overlay.clip(0, 255).astype(np.uint8)

out_path  = REPO_ROOT / "sam3_result.png"
mask_path = REPO_ROOT / "sam3_mask.png"
cv2.imwrite(str(out_path),  overlay)
cv2.imwrite(str(mask_path), combined_mask)
print(f"\nSaved: {out_path}")
print(f"Saved: {mask_path}")
