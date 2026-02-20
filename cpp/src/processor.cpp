// processor.cpp
// Author: Jason Hughes
// Date:   2026
//
// SAM3 image + text pre-processor and mask post-processor.

#include "sam3/processor.hpp"

#include <stdexcept>
#include <algorithm>

namespace Sam3
{

// ---------------------------------------------------------------------------
// Sam3Parameters
// ---------------------------------------------------------------------------

Sam3Parameters::Sam3Parameters(const std::string& yaml_path)
{
    YAML::Node cfg;
    try
    {
        cfg = YAML::LoadFile(yaml_path);
    }
    catch (const YAML::BadFile& e)
    {
        throw std::runtime_error("Sam3Parameters: cannot open " + yaml_path);
    }

    image_size     = cfg["image_size"].as<int>(1008);
    text_padding   = cfg["text_padding"].as<int>(77);
    score_threshold= cfg["score_threshold"].as<float>(0.5f);
    mask_threshold = cfg["mask_threshold"].as<float>(0.5f);

    auto mean_node = cfg["image_mean"];
    auto std_node  = cfg["image_std"];

    for (int i = 0; i < 3; ++i)
    {
        image_mean[i] = mean_node[i].as<float>(0.5f);
        image_std[i]  = std_node[i].as<float>(0.5f);
    }
}

// ---------------------------------------------------------------------------
// Sam3ModelInputs
// ---------------------------------------------------------------------------

Sam3ModelInputs Sam3ModelInputs::fromImageAndText(const at::Tensor& image,
                                                   const at::Tensor& ids,
                                                   const at::Tensor& mask,
                                                   int orig_h, int orig_w)
{
    Sam3ModelInputs inp;
    inp.pixel_values   = image;
    inp.input_ids      = ids;
    inp.attention_mask = mask;
    inp.orig_height    = orig_h;
    inp.orig_width     = orig_w;
    return inp;
}

// ---------------------------------------------------------------------------
// Sam3Processor
// ---------------------------------------------------------------------------

Sam3Processor::Sam3Processor(const std::string& yaml_path,
                             const std::string& merges_path,
                             const std::string& vocab_path)
    : params_(yaml_path),
      tokenizer_(merges_path, vocab_path)
{}

// ---------------------------------------------------------------------------

Sam3ModelInputs Sam3Processor::process(const cv::Mat& image, const std::string& text) const
{
    if (image.empty())
        throw std::runtime_error("Sam3Processor::process: empty image.");

    int orig_h = image.rows;
    int orig_w = image.cols;

    at::Tensor pixel_values = processImage(image);

    at::Tensor attention_mask;
    at::Tensor input_ids = processText(text, attention_mask);

    return Sam3ModelInputs::fromImageAndText(
        pixel_values, input_ids, attention_mask, orig_h, orig_w);
}

// ---------------------------------------------------------------------------

at::Tensor Sam3Processor::processImage(const cv::Mat& image) const
{
    int sz = params_.image_size;

    // Resize to (sz, sz) directly (same as HuggingFace Sam3ImageProcessorFast)
    cv::Mat resized = resizeImage(image, sz, sz);

    // Normalize: BGR->RGB, /255, (x-mean)/std
    std::vector<float> mean(params_.image_mean, params_.image_mean + 3);
    std::vector<float> std (params_.image_std,  params_.image_std  + 3);
    cv::Mat normalized = normalizeImage(resized, mean, std);

    // HxWxC float32 -> (1, C, H, W) tensor
    return cvToTensor(normalized);
}

// ---------------------------------------------------------------------------

at::Tensor Sam3Processor::processText(const std::string& text, at::Tensor& attention_mask_out) const
{
    std::vector<int> ids = tokenizer_.tokenize(text);

    // Use actual token length (no padding) to match HuggingFace processor behaviour.
    // The JIT models were traced with unpadded sequences.
    int real_len = std::min(static_cast<int>(ids.size()), params_.text_padding);

    std::vector<int64_t> id_vec(real_len);
    std::vector<int64_t> mask_vec(real_len, 1LL);   // all real tokens → mask = 1
    for (int i = 0; i < real_len; ++i)
        id_vec[i] = static_cast<int64_t>(ids[i]);

    at::Tensor id_tensor   = torch::tensor(id_vec,   torch::kInt64).unsqueeze(0);  // (1, T)
    at::Tensor mask_tensor = torch::tensor(mask_vec, torch::kInt64).unsqueeze(0);  // (1, T)

    attention_mask_out = mask_tensor;
    return id_tensor;
}

// ---------------------------------------------------------------------------
// Post-processing
// ---------------------------------------------------------------------------

std::vector<cv::Mat> Sam3Processor::postProcess(const at::Tensor& pred_masks, const at::Tensor& pred_logits, const at::Tensor& presence_logits, int orig_h, int orig_w) const
{
    // Mirrors HuggingFace post_process_instance_segmentation():
    //   scores  = sigmoid(pred_logits) * sigmoid(presence_logits)  -> (Q,)
    //   keep    = scores > score_threshold
    //   for each kept mask: bilinear upsample to (orig_h, orig_w), binarize

    at::Tensor scores = (torch::sigmoid(pred_logits) *
                         torch::sigmoid(presence_logits)).squeeze(0);  // (Q,)
    at::Tensor masks  = torch::sigmoid(pred_masks).squeeze(0);          // (Q, H_m, W_m)

    at::Tensor keep   = (scores > params_.score_threshold).nonzero().squeeze(1); // (K,)

    // Fallback: keep best query if nothing passes threshold
    if (keep.numel() == 0)
        keep = torch::tensor({scores.argmax().item<int64_t>()});

    at::Tensor sel_masks = masks.index_select(0, keep);  // (K, H_m, W_m)

    // Bilinear upsample: (1, K, H_m, W_m) -> (1, K, orig_h, orig_w)
    at::Tensor sel_4d = sel_masks.unsqueeze(0);
    at::Tensor up = torch::nn::functional::interpolate(
        sel_4d,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{orig_h, orig_w})
            .mode(torch::kBilinear)
            .align_corners(false)
    ).squeeze(0);  // (K, orig_h, orig_w)

    // Binarize and convert each instance to CV_8UC1
    std::vector<cv::Mat> result;
    result.reserve(keep.numel());
    for (int64_t i = 0; i < keep.numel(); ++i)
    {
        at::Tensor bin = (up[i] > params_.mask_threshold).to(torch::kFloat32);
        cv::Mat m = tensorToCv(bin);          // CV_32FC1
        m.convertTo(m, CV_8UC1, 255.0);
        result.push_back(m.clone());
    }

    return result;
}

}  // namespace Sam3
