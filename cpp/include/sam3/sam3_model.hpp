// sam3_model.hpp
// Author: Jason Hughes
// Date:   2026
//
// TorchScript JIT wrappers for the four SAM3 sub-models.
// Follows the same pattern as jhughes50/clipper.

#pragma once

#include <string>
#include <vector>
#include <torch/script.h>
#include <torch/torch.h>
#include <glog/logging.h>

namespace Sam3
{

class Sam3ModelBase
{
public: 
    explicit Sam3ModelBase(const std::string& model_path);
    
    torch::ScalarType getDType() const;

protected:
    torch::jit::script::Module module_;
    torch::Device device_;
    torch::ScalarType dtype_;   ///< kHalf on CUDA, kFloat32 on CPU

    static torch::Device selectDevice();

    /// Move tensor to the model's device and dtype.
    at::Tensor toDevice(const at::Tensor& t) const;
    /// Move tensor back to CPU float32 (for downstream C++ / OpenCV code).
    at::Tensor toHost(const at::Tensor& t) const;
};

/// forward(pixel_values) -> (fpn_0, fpn_1, fpn_2, fpn_3,
///                           fpn_pos_0, fpn_pos_1, fpn_pos_2, fpn_pos_3)
class Sam3ImageEncoder : public Sam3ModelBase
{
public:
    explicit Sam3ImageEncoder(const std::string& model_path) : Sam3ModelBase(model_path) {}

    /// @param pixel_values (1, 3, H, W) float32
    /// @returns 8-element vector: 4 FPN feature maps then 4 position encodings
    std::vector<at::Tensor> forward(const at::Tensor& pixel_values);
};

/// Runs CLIP text encoder + SAM3 linear projection.
///
/// forward(input_ids, attention_mask) -> text_embeds  (1, 256)
class Sam3TextEncoder : public Sam3ModelBase
{
public:
    explicit Sam3TextEncoder(const std::string& model_path)
        : Sam3ModelBase(model_path) {}

    /// @param input_ids      (1, 77) int64
    /// @param attention_mask (1, 77) int64
    /// @returns text_embeds  (1, 256) float32
    at::Tensor forward(const at::Tensor& input_ids,
                       const at::Tensor& attention_mask);

    /// Cache text embeddings for a set of prompts (call once, reuse).
    void setText(const at::Tensor& input_ids,
                 const at::Tensor& attention_mask);

    const at::Tensor& getTextEmbeds()   const { return text_embeds_; }
    const at::Tensor& getAttentionMask() const { return stored_mask_; }

private:
    at::Tensor text_embeds_;
    at::Tensor stored_mask_;
    bool       cached_ = false;
};

/// Runs the geometry encoder for box prompts.
/// Not called for text-only inference.
///
/// forward(input_boxes, input_boxes_labels, roi_features) -> prompt_embeds
class Sam3PromptEncoder : public Sam3ModelBase
{
public:
    explicit Sam3PromptEncoder(const std::string& model_path)
        : Sam3ModelBase(model_path) {}

    /// @param input_boxes        (1, N, 4) float32 – normalized (cx,cy,w,h)
    /// @param input_boxes_labels (1, N)    int64
    /// @param roi_features       (1, C, H, W) float32 – highest-res FPN scale
    /// @returns prompt_embeds    (1, N, D) float32
    at::Tensor forward(const at::Tensor& input_boxes,
                       const at::Tensor& input_boxes_labels,
                       const at::Tensor& roi_features);
};

// ---------------------------------------------------------------------------
// Mask decoder  (mask_decoder.pt)
// ---------------------------------------------------------------------------

/// Runs DETR encoder + DETR decoder + mask head + dot-product scoring.
///
/// forward(fpn_0..3, fpn_pos_0..3, text_embeds, attention_mask)
///      -> (pred_masks, pred_logits, presence_logits)
class Sam3MaskDecoder : public Sam3ModelBase
{
public:
    explicit Sam3MaskDecoder(const std::string& model_path) : Sam3ModelBase(model_path) {}

    /// @param vision_features  8-element vector from Sam3ImageEncoder::forward()
    ///                         (fpn_0..3 then fpn_pos_0..3)
    /// @param text_embeds      (1, T, 256) float32
    /// @param attention_mask   (1, T)      int64
    /// @param original_sizes   (1, 2)      int64  – original image [H, W]
    /// @returns tuple: (pred_masks, pred_logits, presence_logits)
    std::tuple<at::Tensor, at::Tensor, at::Tensor>
    forward(const std::vector<at::Tensor>& vision_features,
            const at::Tensor& text_embeds,
            const at::Tensor& attention_mask,
            const at::Tensor& original_sizes);
};

// ---------------------------------------------------------------------------
// Top-level orchestrator
// ---------------------------------------------------------------------------

struct Sam3Output
{
    at::Tensor pred_masks;       // (1, Q, H, W) raw mask logits
    at::Tensor pred_logits;      // (1, Q) classification logits
    at::Tensor presence_logits;  // (1, 1) scene presence logit
};

/// Pre-processed text for one class / prompt.
struct Sam3TextInputs
{
    at::Tensor input_ids;       ///< (1, T) int64
    at::Tensor attention_mask;  ///< (1, T) int64
};

/// Convenience class that owns all four sub-models and orchestrates inference.
class Sam3
{
public:
    /// @param models_dir  Directory containing the four .pt files
    explicit Sam3(const std::string& models_dir);

    /// Pre-compute and cache vision features for a given image.
    /// orig_h / orig_w are the image dimensions BEFORE the 1008×1008 resize;
    /// they are forwarded to the mask decoder as original_sizes.
    /// Call this once per image, then call inference() for each text prompt.
    void setImage(const at::Tensor& pixel_values, int orig_h, int orig_w);

    /// Pre-compute and cache text features for a given prompt.
    void setText(const at::Tensor& input_ids,
                 const at::Tensor& attention_mask);

    /// Run full forward pass (image + text must have been set first).
    Sam3Output inference();

    /// Encode text and decode masks for each class, reusing the cached image
    /// features.  setImage() must be called before this.
    /// Returns one Sam3Output per entry in text_inputs.
    std::vector<Sam3Output> inferenceMultiClass(const std::vector<Sam3TextInputs>& text_inputs);

    /// Single-call convenience: encode image + text and decode masks.
    Sam3Output operator()(const at::Tensor& pixel_values,
                          const at::Tensor& input_ids,
                          const at::Tensor& attention_mask,
                          int orig_h, int orig_w);

private:
    Sam3ImageEncoder  image_encoder_;
    Sam3TextEncoder   text_encoder_;
    std::unique_ptr<Sam3PromptEncoder> prompt_encoder_;  ///< null if .pt not present
    Sam3MaskDecoder   mask_decoder_;
    
    torch::ScalarType dtype_;
    std::vector<at::Tensor> vision_features_;
    int  orig_h_       = 0;
    int  orig_w_       = 0;
    bool image_cached_ = false;
};

}  // namespace Sam3
