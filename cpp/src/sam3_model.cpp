// sam3_model.cpp
// Author: Jason Hughes
// Date:   2026
//
// TorchScript JIT wrapper implementations for SAM3.

#include "sam3/sam3_model.hpp"

#include <filesystem>
#include <stdexcept>

namespace Sam3
{

torch::Device Sam3ModelBase::selectDevice()
{
    // Allow runtime override via SAM3_DEVICE env var ("cpu" to force CPU).
    const char* env = std::getenv("SAM3_DEVICE");
    if (env && std::string(env) == "cpu")
    {
        LOG(INFO) << "SAM3: using CPU (SAM3_DEVICE override).";
        return torch::Device(torch::kCPU);
    }

    if (torch::cuda::is_available())
    {
        LOG(INFO) << "SAM3: using CUDA.";
        return torch::Device(torch::kCUDA);
    }

    LOG(INFO) << "SAM3: CUDA not available, falling back to CPU.";
    return torch::Device(torch::kCPU);
}

Sam3ModelBase::Sam3ModelBase(const std::string& model_path): device_(selectDevice()), dtype_(torch::kBFloat16)
{
    if (!std::filesystem::exists(model_path))
        throw std::runtime_error("Sam3ModelBase: model file not found: " + model_path);

    try {
        LOG(INFO) << "[SAM3] Using dtype: " << dtype_;
        module_ = torch::jit::load(model_path);
        module_.eval();
        module_.to(device_);
        //module_.to(dtype_);
    }
    catch (const c10::Error& e) {
        throw std::runtime_error("[Sam3ModelBase]: failed to load " + model_path + "\n" + e.what());
    }
}

at::Tensor Sam3ModelBase::toDevice(const at::Tensor& t) const
{
    at::Tensor moved = t.to(device_);

    if (at::isFloatingType(t.scalar_type())) {
        return moved.to(dtype_);
    }

    return moved;
}

at::Tensor Sam3ModelBase::toHost(const at::Tensor& t) const
{
    return t.to(torch::kCPU).to(torch::kFloat32);
}

torch::ScalarType Sam3ModelBase::getDType() const
{
    return dtype_;
}

std::vector<at::Tensor> Sam3ImageEncoder::forward(const at::Tensor& pixel_values)
{
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs{ toDevice(pixel_values) };
    auto output = module_.forward(inputs);

    // The traced module returns a tuple of 8 tensors:
    // fpn_0, fpn_1, fpn_2, fpn_3, fpn_pos_0, fpn_pos_1, fpn_pos_2, fpn_pos_3
    auto tuple = output.toTuple();
    std::vector<at::Tensor> result;
    result.reserve(tuple->elements().size());
    for (const auto& elem : tuple->elements())
        result.push_back(elem.toTensor());  // keep on device for mask decoder

    return result;
}


at::Tensor Sam3TextEncoder::forward(const at::Tensor& input_ids, const at::Tensor& attention_mask)
{
    torch::NoGradGuard no_grad;
    at::Tensor ninput_ids = toDevice(input_ids);
    at::Tensor nattention_mask = toDevice(attention_mask);
    std::vector<torch::jit::IValue> inputs{
        ninput_ids,       // int64 – stays int, moved to device
        nattention_mask   // int64 – stays int, moved to device
    };
    return module_.forward(inputs).toTensor();  // keep on device for mask decoder
}

void Sam3TextEncoder::setText(const at::Tensor& input_ids,
                              const at::Tensor& attention_mask)
{
    text_embeds_  = forward(input_ids, attention_mask);
    stored_mask_  = toDevice(attention_mask);
    cached_       = true;
}

at::Tensor Sam3PromptEncoder::forward(const at::Tensor& input_boxes,
                                      const at::Tensor& input_boxes_labels,
                                      const at::Tensor& roi_features)
{
    std::vector<torch::jit::IValue> inputs{
        toDevice(input_boxes),
        toDevice(input_boxes_labels),
        toDevice(roi_features)
    };
    return module_.forward(inputs).toTensor();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
Sam3MaskDecoder::forward(const std::vector<at::Tensor>& vision_features,
                         const at::Tensor& text_embeds,
                         const at::Tensor& attention_mask,
                         const at::Tensor& original_sizes)
{
    // vision_features has 8 tensors: fpn_0..3, fpn_pos_0..3
    torch::NoGradGuard no_grad;
    if (vision_features.size() != 8)
        throw std::runtime_error(
            "Sam3MaskDecoder::forward: expected 8 vision feature tensors, got "
            + std::to_string(vision_features.size()));

    std::vector<torch::jit::IValue> inputs;
    inputs.reserve(11);
    for (const auto& t : vision_features)
        inputs.push_back(t);  // already on device (from ImageEncoder)
    inputs.push_back(text_embeds);      // already on device (from TextEncoder)
    inputs.push_back(attention_mask);   // already on device (from TextEncoder)
    inputs.push_back(original_sizes.to(device_));  // int64 – no dtype cast

    auto output = module_.forward(inputs).toTuple();
    at::Tensor pred_masks      = toHost(output->elements()[0].toTensor());
    at::Tensor pred_logits     = toHost(output->elements()[1].toTensor());
    at::Tensor presence_logits = toHost(output->elements()[2].toTensor());

    return {pred_masks, pred_logits, presence_logits};
}

Sam3::Sam3(const std::string& models_dir)
    : image_encoder_(models_dir + "/image_encoder_bf16.pt"),
      text_encoder_ (models_dir + "/text_encoder_bf16.pt"),
      mask_decoder_ (models_dir + "/mask_decoder_bf16.pt")
{
    dtype_ = image_encoder_.getDType();
    const std::string prompt_path = models_dir + "/prompt_encoder.pt";
    if (std::filesystem::exists(prompt_path))
        prompt_encoder_ = std::make_unique<Sam3PromptEncoder>(prompt_path);
    else
        LOG(INFO) << "SAM3: prompt_encoder.pt not found – box prompts disabled.";
}

void Sam3::setImage(const at::Tensor& pixel_values, int orig_h, int orig_w)
{
    vision_features_.clear();
    vision_features_ = image_encoder_.forward(pixel_values);
    orig_h_          = orig_h;
    orig_w_          = orig_w;
    image_cached_    = true;
}

void Sam3::setText(const at::Tensor& input_ids,
                   const at::Tensor& attention_mask)
{
    text_encoder_.setText(input_ids, attention_mask);
}

Sam3Output Sam3::inference()
{
    torch::NoGradGuard no_grad;
    if (!image_cached_)
        throw std::runtime_error("Sam3::inference: call setImage() first.");
    if (!text_encoder_.getTextEmbeds().defined())
        throw std::runtime_error("Sam3::inference: call setText() first.");

    at::Tensor original_sizes = torch::tensor(
        {{static_cast<int64_t>(orig_h_), static_cast<int64_t>(orig_w_)}},
        torch::kInt64);

    auto [pred_masks, pred_logits, presence_logits] =
        mask_decoder_.forward(vision_features_,
                              text_encoder_.getTextEmbeds(),
                              text_encoder_.getAttentionMask(),
                              original_sizes);

    return Sam3Output{ pred_masks, pred_logits, presence_logits };
}

std::vector<Sam3Output> Sam3::inferenceMultiClass(const std::vector<Sam3TextInputs>& text_inputs)
{
    torch::NoGradGuard no_grad;
    if (!image_cached_)
        throw std::runtime_error("Sam3::inferenceMultiClass: call setImage() first.");

    at::Tensor original_sizes = torch::tensor(
        {{static_cast<int64_t>(orig_h_), static_cast<int64_t>(orig_w_)}},
        torch::kInt64);

    std::vector<Sam3Output> results;
    results.reserve(text_inputs.size());

    for (const auto& ti : text_inputs)
    {
        // setText caches the text embeds and moves the mask to device.
        text_encoder_.setText(ti.input_ids, ti.attention_mask);

        auto [pred_masks, pred_logits, presence_logits] =
            mask_decoder_.forward(vision_features_,
                                  text_encoder_.getTextEmbeds(),
                                  text_encoder_.getAttentionMask(),
                                  original_sizes);

        results.push_back(Sam3Output{pred_masks, pred_logits, presence_logits});
    }

    return results;
}

Sam3Output Sam3::operator()(const at::Tensor& pixel_values,
                             const at::Tensor& input_ids,
                             const at::Tensor& attention_mask,
                             int orig_h, int orig_w)
{
    setImage(pixel_values, orig_h, orig_w);
    setText(input_ids, attention_mask);
    return inference();
}

}  // namespace Sam3
