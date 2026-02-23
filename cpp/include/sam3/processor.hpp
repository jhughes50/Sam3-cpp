// processor.hpp
// Author: Jason Hughes
// Date:   2026
//
// SAM3 image and text pre/post-processor.
// Mirrors the HuggingFace Sam3Processor + Sam3ImageProcessorFast in C++.

#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "sam3/tokenizer.hpp"
#include "sam3/processor_mixins.hpp"
#include "sam3/sam3_model.hpp"

namespace Sam3
{

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct Sam3Parameters
{
    int   image_size;        ///< Target square image size (default 1008)
    float image_mean[3];     ///< Per-channel mean  {R, G, B}
    float image_std[3];      ///< Per-channel std   {R, G, B}
    int   text_padding;      ///< Max token length (default 77)
    float score_threshold;   ///< Minimum final score to keep a mask
    float mask_threshold;    ///< Sigmoid threshold for binary mask

    explicit Sam3Parameters(const std::string& yaml_path);
};

// ---------------------------------------------------------------------------
// Model inputs
// ---------------------------------------------------------------------------

struct Sam3ModelInputs
{
    at::Tensor pixel_values;   ///< (1, 3, H, W) float32
    at::Tensor input_ids;      ///< (1, text_padding) int64
    at::Tensor attention_mask; ///< (1, text_padding) int64

    /// Original image dimensions before resize (for post-process rescaling)
    int orig_height;
    int orig_width;

    static Sam3ModelInputs fromImageAndText(const at::Tensor& image,
                                            const at::Tensor& ids,
                                            const at::Tensor& mask,
                                            int orig_h, int orig_w);
};

// ---------------------------------------------------------------------------
// Processor
// ---------------------------------------------------------------------------

class Sam3Processor
{
public:
    /// @param yaml_path    Path to sam3.yaml
    /// @param merges_path  Path to merges.txt
    /// @param vocab_path   Path to vocab.json
    Sam3Processor(const std::string& yaml_path,
                  const std::string& merges_path,
                  const std::string& vocab_path);

    /// Pre-process a raw BGR image and a text prompt string.
    Sam3ModelInputs process(const cv::Mat& image, const std::string& text) const;

    /// Pre-process image only (pixel_values + orig dimensions; text fields unset).
    Sam3ModelInputs processImageOnly(const cv::Mat& image) const;

    /// Pre-process a list of text prompts, one Sam3TextInputs per entry.
    std::vector<Sam3TextInputs> processTexts(const std::vector<std::string>& texts) const;

    /// Post-process raw model outputs into per-instance binary masks.
    /// Mirrors HuggingFace Sam3Processor.post_process_instance_segmentation().
    ///
    /// @param pred_masks      (1, num_queries, H_mask, W_mask) – raw logits
    /// @param pred_logits     (1, num_queries)
    /// @param presence_logits (1, 1)
    /// @param orig_h          Original image height
    /// @param orig_w          Original image width
    /// @returns               One CV_8UC1 binary mask per instance (score > threshold)
    std::vector<cv::Mat> postProcess(const at::Tensor& pred_masks,
                                     const at::Tensor& pred_logits,
                                     const at::Tensor& presence_logits,
                                     int orig_h, int orig_w) const;

    const Sam3Parameters& params() const { return params_; }

private:
    at::Tensor processImage(const cv::Mat& image) const;
    at::Tensor processText (const std::string& text,
                            at::Tensor& attention_mask_out) const;

    Sam3Parameters  params_;
    CLIPTokenizer   tokenizer_;
};

}  // namespace Sam3
