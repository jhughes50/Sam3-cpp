#!/usr/bin/env python3
"""
Export SAM3 (Segment Anything Model 3) image components as TorchScript JIT modules.

Exports four separate modules for C++ inference:
  - image_encoder.pt  : ViT backbone + FPN neck -> (fpn_0..3, fpn_pos_0..3)
  - text_encoder.pt   : CLIP text encoder + SAM3 projection -> text_embeds
  - prompt_encoder.pt : Geometry encoder for box prompts -> geometry_embeds
  - mask_decoder.pt   : DETR encoder + decoder + mask head + scoring -> masks, logits

Usage:
    HF_KEY=<your_token> python3 export.py

Author: Jason Hughes
Date:   2026
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

try:
    from transformers import Sam3Model, Sam3Processor
except ImportError:
    print("Error: transformers>=4.47.0 is required for SAM3 support.")
    print("  pip install 'transformers>=4.47.0'")
    sys.exit(1)

MODEL_ID   = "facebook/sam3"
HF_KEY     = os.environ.get("HF_KEY")
REPO_ROOT  = Path(__file__).parent.parent
MODELS_DIR = REPO_ROOT / "cpp" / "models"
CONFIG_DIR = REPO_ROOT / "cpp" / "config"

# Force CPU – GPU is optional and the export works identically on CPU.
# The C++ runtime will pick up CUDA automatically if available at inference time.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
EXPORT_DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------

class ImageEncoderWrapper(torch.nn.Module):
    """
    Wraps Sam3VisionModel (ViT + FPN neck).

    Input:
        pixel_values: (1, 3, H, W) float32, normalized

    Output (8-tuple):
        fpn_0..3    : FPN feature maps at 4 scales (B, C, H_i, W_i)
        fpn_pos_0..3: Corresponding position encodings  (B, C', H_i, W_i)
    """
    def __init__(self, model: Sam3Model):
        super().__init__()
        self.vision_encoder = model.vision_encoder

    def forward(
        self, pixel_values: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        output = self.vision_encoder(pixel_values)
        fpn = output.fpn_hidden_states      # tuple of 4 (B, C, H, W)
        pos = output.fpn_position_encoding  # tuple of 4 (B, C', H, W)
        return fpn[0], fpn[1], fpn[2], fpn[3], pos[0], pos[1], pos[2], pos[3]


class TextEncoderWrapper(torch.nn.Module):
    """
    Wraps CLIPTextModelWithProjection + SAM3 text_projection.

    CLIPTextModelOutput has NO pooler_output — it has last_hidden_state (B, T, 1024)
    and text_embeds (B, 512, CLIP's own projection).
    SAM3's text_projection is Linear(1024, 256), so it takes last_hidden_state.

    Projecting the full sequence gives (B, T, 256) which the DETR encoder/decoder
    use for cross-attention (with text_mask to ignore padding).

    Input:
        input_ids     : (1, T) int64
        attention_mask: (1, T) int64

    Output:
        text_embeds: (1, T, 256) float32 - per-token projected text features
    """
    def __init__(self, model: Sam3Model):
        super().__init__()
        self.text_encoder    = model.text_encoder    # CLIPTextModelWithProjection
        self.text_projection = model.text_projection  # Linear(1024, 256)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        output      = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state: (B, T, 1024) -> project all tokens -> (B, T, 256)
        text_embeds = self.text_projection(output.last_hidden_state)
        return text_embeds


class PromptEncoderWrapper(torch.nn.Module):
    """
    Wraps Sam3GeometryEncoder for visual (box) prompts.

    For text-only inference this module is not called; it is exported for
    completeness so the C++ library can support box prompts in future.

    Actual Sam3GeometryEncoder.forward params (discovered at runtime):
        box_embeddings : (1, N, D) float32 – pre-computed box position embeddings
        box_mask       : (1, N)    bool    – True where box is padding
        box_labels     : (1, N)    int64   – 1=positive, 0=negative
        img_feats      : (1, C, H, W) float32 – FPN image features (for ROI pooling)
        img_pos_embeds : (1, C, H, W) float32 – FPN position encodings

    Output:
        prompt_embeds: (1, N, D) float32
    """
    def __init__(self, model: Sam3Model):
        super().__init__()
        self.geometry_encoder = model.geometry_encoder

    def forward(
        self,
        box_embeddings : torch.Tensor,
        box_mask       : torch.Tensor,
        box_labels     : torch.Tensor,
        img_feats      : torch.Tensor,
        img_pos_embeds : torch.Tensor,
    ) -> torch.Tensor:
        output = self.geometry_encoder(
            box_embeddings=box_embeddings,
            box_mask=box_mask,
            box_labels=box_labels,
            img_feats=img_feats,
            img_pos_embeds=img_pos_embeds,
        )
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if isinstance(output, torch.Tensor):
            return output
        return output[0]


class MaskDecoderWrapper(torch.nn.Module):
    """
    Wraps DETR encoder + DETR decoder + mask head + dot-product scoring.

    Designed for text-only inference (no geometry prompts).

    Inputs (11 tensors):
        fpn_0..3      : FPN features from image encoder (4 scales)
        fpn_pos_0..3  : FPN position encodings from image encoder
        text_embeds   : (1, T, 256) from text encoder
        attention_mask: (1, T) int64
        original_sizes: (1, 2) int64 – original image [H, W] before resize

    Outputs (3 tensors):
        pred_masks     : (1, num_queries, H_mask, W_mask)
        pred_logits    : (1, num_queries) – classification scores (pre-sigmoid)
        presence_logits: (1, 1) – scene-level presence score (pre-sigmoid)

    Note: the DETR encoder only receives fpn[2] (72×72), the single scale the
    full SAM3 model uses. All 4 FPN levels are forwarded to the mask head for
    multi-scale upsampling.
    """
    def __init__(self, model: Sam3Model):
        super().__init__()
        self.detr_encoder      = model.detr_encoder
        self.detr_decoder      = model.detr_decoder
        self.mask_decoder      = model.mask_decoder
        self.dot_product_scoring = model.dot_product_scoring

    def forward(
        self,
        fpn_0:    torch.Tensor,
        fpn_1:    torch.Tensor,
        fpn_2:    torch.Tensor,
        fpn_3:    torch.Tensor,
        fpn_pos_0: torch.Tensor,
        fpn_pos_1: torch.Tensor,
        fpn_pos_2: torch.Tensor,
        fpn_pos_3: torch.Tensor,
        text_embeds:    torch.Tensor,
        attention_mask: torch.Tensor,
        original_sizes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        vision_features   = [fpn_0, fpn_1, fpn_2, fpn_3]

        # DETR encoder expects a single 72×72 scale (fpn[2] / pos[2])
        enc_out = self.detr_encoder(
            vision_features=[fpn_2],
            text_features=text_embeds,
            vision_pos_embeds=[fpn_pos_2],
            text_mask=attention_mask.bool(),
            original_sizes=original_sizes,
        )

        # --- DETR decoder: object queries ---
        dec_out = self.detr_decoder(
            vision_features=enc_out.last_hidden_state,
            text_features=text_embeds,
            vision_pos_encoding=enc_out.pos_embeds_flattened,
            text_mask=attention_mask,
            spatial_shapes=enc_out.spatial_shapes,
        )

        # --- Dot-product scoring: (num_layers, B, num_queries, 1) ---
        scores = self.dot_product_scoring(
            decoder_hidden_states=dec_out.intermediate_hidden_states,
            text_features=text_embeds,
            text_mask=attention_mask,
        )
        # Take last-layer scores and squeeze trailing dim -> (B, num_queries)
        n_layers    = scores.shape[0]
        pred_logits = scores[n_layers - 1, :, :, 0]

        # Presence logit from decoder presence token.
        # dec_out.presence_logits may be (num_layers, B, 1) – take last layer -> (B, 1)
        presence_logits = dec_out.presence_logits
        if presence_logits.dim() == 3:
            presence_logits = presence_logits[-1]

        # --- Mask decoder: pixel-level masks ---
        # Use last-layer decoder queries
        n_int        = dec_out.intermediate_hidden_states.shape[0]
        last_queries = dec_out.intermediate_hidden_states[n_int - 1]  # (B, Q, D)

        # backbone_features: fpn[0..2] only (288, 144, 72 px) – fpn[3] (36 px) excluded
        # prompt_features / prompt_mask: text context fed into the mask head
        mask_out = self.mask_decoder(
            decoder_queries=last_queries,
            backbone_features=[fpn_0, fpn_1, fpn_2],
            encoder_hidden_states=enc_out.last_hidden_state,
            prompt_features=text_embeds,
            prompt_mask=attention_mask.bool(),
            original_sizes=original_sizes,
        )

        return mask_out.pred_masks, pred_logits, presence_logits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_tokenizer_files(processor, config_dir: Path) -> None:
    """Save CLIP BPE merges and vocabulary to config directory."""
    tokenizer = processor.tokenizer

    # --- vocab.json ---
    vocab = tokenizer.get_vocab()
    with open(config_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"  vocab.json  : {len(vocab)} tokens")

    # --- merges.txt ---
    lines = ["#version: 0.2"]
    if hasattr(tokenizer, "bpe_ranks"):
        # older transformers: dict {(a, b): rank}
        sorted_merges = sorted(tokenizer.bpe_ranks.items(), key=lambda x: x[1])
        for (a, b), _ in sorted_merges:
            lines.append(f"{a} {b}")
    elif hasattr(tokenizer, "_merges"):
        # newer transformers: list of "a b" strings
        for m in tokenizer._merges:
            if isinstance(m, (list, tuple)):
                lines.append(f"{m[0]} {m[1]}")
            else:
                lines.append(str(m))
    elif hasattr(tokenizer, "merges"):
        lines.extend(tokenizer.merges)
    else:
        print("  WARNING: could not extract BPE merges from tokenizer.")

    with open(config_dir / "merges.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  merges.txt  : {len(lines) - 1} merges")


def save_config(processor, config_dir: Path) -> dict:
    """Save image pre-processing config to YAML; return config dict."""
    img_proc = processor.image_processor

    image_mean = list(img_proc.image_mean) if hasattr(img_proc, "image_mean") else [0.5, 0.5, 0.5]
    image_std  = list(img_proc.image_std)  if hasattr(img_proc, "image_std")  else [0.5, 0.5, 0.5]

    size_dict  = getattr(img_proc, "size", {"height": 1008, "width": 1008})
    image_size = size_dict.get("height", 1008) if isinstance(size_dict, dict) else int(size_dict)

    config = {
        "image_size"  : int(image_size),
        "image_mean"  : [float(x) for x in image_mean],
        "image_std"   : [float(x) for x in image_std],
        "text_padding": 77,
    }
    with open(config_dir / "sam3.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"  sam3.yaml   : size={image_size}, mean={image_mean}, std={image_std}")
    return config


def trace_and_save(wrapper: torch.nn.Module,
                   example_inputs: tuple,
                   save_path: Path,
                   name: str) -> bool:
    """Trace a wrapper module and save as TorchScript."""
    wrapper.eval()
    with torch.no_grad():
        try:
            traced = torch.jit.trace(wrapper, example_inputs, strict=False)
            torch.jit.save(traced, str(save_path))
            print(f"  {name}: saved -> {save_path.name}")
            return True
        except Exception as e:
            print(f"  {name}: torch.jit.trace FAILED ({e})")
            print(f"  {name}: trying torch.jit.script ...")
            try:
                scripted = torch.jit.script(wrapper)
                torch.jit.save(scripted, str(save_path))
                print(f"  {name}: saved (scripted) -> {save_path.name}")
                return True
            except Exception as e2:
                print(f"  {name}: FAILED to export ({e2})")
                return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(export_dtype: torch.dtype = EXPORT_DTYPE) -> None:
    print(f"Loading SAM3 ({MODEL_ID}) on {DEVICE} with dtype {export_dtype}...")
    
    load_kwargs = {"token": HF_KEY} if HF_KEY else {}
    model = Sam3Model.from_pretrained(MODEL_ID, **load_kwargs)
    processor = Sam3Processor.from_pretrained(MODEL_ID, **load_kwargs)
    
    # -----------------------------------------------------------------------
    # Move model to device and correct precision
    # -----------------------------------------------------------------------
    model = model.to(device=DEVICE, dtype=export_dtype).eval()

    print("\nTop-level sub-modules:")
    for name, module in model.named_children():
        p = sum(p.numel() for p in module.parameters()) / 1e6
        print(f"  {name:30s} {type(module).__name__} ({p:.1f}M params)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Config + tokenizer files
    print("\nSaving config files:")
    save_tokenizer_files(processor, CONFIG_DIR)
    config = save_config(processor, CONFIG_DIR)
    
    # -----------------------------------------------------------------------
    # Sanity Check (Using correct dtype for inputs)
    # -----------------------------------------------------------------------
    print("\nRunning sanity check (full model) ...")
    from PIL import Image as PILImage
    test_img_path = REPO_ROOT / "test.png"
    
    if test_img_path.exists():
        img = PILImage.open(test_img_path).convert("RGB")
        w, h = img.size
        img = img.resize((w // 4, h // 4), PILImage.BILINEAR)
    else:
        rng = np.random.default_rng(42)
        img = PILImage.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8))

    # Processor returns FP32 by default, we cast pixel_values to our export_dtype
    inp = processor(images=img, text="road", return_tensors="pt").to(DEVICE)
    pixel_values = inp.pixel_values.to(export_dtype) 
    input_ids = inp.input_ids
    attention_mask = inp.attention_mask

    with torch.no_grad():
        # Pass the casted pixel_values explicitly
        out = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        
    print(f"  pred_masks    : {out.pred_masks.shape} ({out.pred_masks.dtype})")

    # -----------------------------------------------------------------------
    # Export: image encoder
    # -----------------------------------------------------------------------
    print(f"\nExporting image_encoder ({export_dtype}) ...")
    image_enc = ImageEncoderWrapper(model).to(device=DEVICE, dtype=export_dtype)
    trace_and_save(image_enc, (pixel_values,),
                   MODELS_DIR / "image_encoder_bf16.pt", "image_encoder")

    # -----------------------------------------------------------------------
    # Export: text encoder
    # -----------------------------------------------------------------------
    print(f"\nExporting text_encoder ({export_dtype}) ...")
    text_enc = TextEncoderWrapper(model).to(device=DEVICE)
    trace_and_save(text_enc, (input_ids, attention_mask),
                   MODELS_DIR / "text_encoder_bf16.pt", "text_encoder")

    # -----------------------------------------------------------------------
    # Export: mask decoder
    # -----------------------------------------------------------------------
    print(f"\nExporting mask_decoder ({export_dtype}) ...")
    with torch.no_grad():
        vis_out = model.vision_encoder(pixel_values)
        fpn = vis_out.fpn_hidden_states
        pos = vis_out.fpn_position_encoding
        
        text_out = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = model.text_projection(text_out.last_hidden_state)

    mask_dec = MaskDecoderWrapper(model).to(device=DEVICE, dtype=export_dtype)
    original_sizes = inp["original_sizes"] # Int64, no cast needed
    
    trace_and_save(
        mask_dec,
        (fpn[0], fpn[1], fpn[2], fpn[3],
         pos[0], pos[1], pos[2], pos[3],
         text_embeds, attention_mask, original_sizes),
        MODELS_DIR / "mask_decoder_bf16.pt",
        "mask_decoder",
    )

    # -----------------------------------------------------------------------
    # End-to-end JIT test (Ensuring JIT inputs match export_dtype)
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"End-to-end JIT test ({export_dtype})")
    print("="*60)

    import cv2
    img_cv = cv2.imread(str(test_img_path))
    h0, w0 = img_cv.shape[:2]
    img_cv = cv2.resize(img_cv, (w0 // 4, h0 // 4), interpolation=cv2.INTER_AREA)

    # Load JIT models
    img_enc_jit = torch.jit.load(str(MODELS_DIR / "image_encoder_bf16.pt"), map_location=DEVICE)
    txt_enc_jit = torch.jit.load(str(MODELS_DIR / "text_encoder_bf16.pt"), map_location=DEVICE)
    mask_dec_jit = torch.jit.load(str(MODELS_DIR / "mask_decoder_bf16.pt"), map_location=DEVICE)

    # Pre-process
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (config["image_size"], config["image_size"]), interpolation=cv2.INTER_LINEAR)
    img_float = img_res.astype(np.float32) / 255.0
    for c in range(3):
        img_float[:, :, c] = (img_float[:, :, c] - config["image_mean"][c]) / config["image_std"][c]
    
    # Final input cast to match the JIT model's expected dtype
    jit_pixel_values = torch.from_numpy(img_float.transpose(2, 0, 1)[np.newaxis]).to(device=DEVICE, dtype=export_dtype)

    with torch.no_grad():
        vision_feats = img_enc_jit(jit_pixel_values)
        text_embeds_jit = txt_enc_jit(input_ids, attention_mask)
        pred_masks, pred_logits, presence_logits = mask_dec_jit(
            *vision_feats, text_embeds_jit, attention_mask, original_sizes
        )

    # Post-process (cast back to float32 for sigmoid/math stability)
    pred_masks = pred_masks.float()
    pred_logits = pred_logits.float()
    presence_logits = presence_logits.float()

    logit_scores   = torch.sigmoid(pred_logits)      # (B, Q)
    presence_score = torch.sigmoid(presence_logits)  # (B, 1)
    final_scores   = (logit_scores * presence_score).squeeze(0)  # (Q,)

    # Find the best mask index
    best_idx   = final_scores.argmax().item()
    best_score = final_scores[best_idx].item()
    print(f"  Best query idx={best_idx}, score={best_score:.4f}")
    best_mask_tensor = torch.sigmoid(pred_masks[0, best_idx]).float() 

    binary_mask_bool = (best_mask_tensor > 0.5)

    binary_mask_np = (binary_mask_bool.cpu().numpy().astype(np.uint8)) * 255

    orig_h, orig_w = img_cv.shape[:2]
    mask_full = cv2.resize(binary_mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    overlay = img_cv.copy()
    color_mask = np.array([0, 255, 0], dtype=np.uint8) # BGR Green
    
    mask_bool = mask_full > 0
    overlay[mask_bool] = color_mask  # Set masked pixels to green
    alpha = 0.4
    result = cv2.addWeighted(overlay, alpha, img_cv, 1 - alpha, 0)
    # --- Save results ---
    out_path = REPO_ROOT / "sam3_result.png"
    mask_path = REPO_ROOT / "sam3_mask.png"
    
    cv2.imwrite(str(out_path), result)
    cv2.imwrite(str(mask_path), mask_full)
    
    print(f"  Result saved to: {out_path}")
    print(f"  Mask saved to:   {mask_path}")

if __name__ == "__main__":
    main()
