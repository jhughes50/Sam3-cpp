// processor_mixins.cpp
// Author: Jason Hughes
// Date:   2026
//
// Low-level image manipulation helpers for SAM3.

#include "sam3/processor_mixins.hpp"

#include <stdexcept>

namespace Sam3
{

cv::Mat normalizeImage(const cv::Mat& image,
                       const std::vector<float>& mean,
                       const std::vector<float>& std)
{
    // Convert BGR -> RGB and scale to [0, 1]
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    // Per-channel normalization: (x - mean) / std
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    for (int c = 0; c < 3; ++c)
        channels[c] = (channels[c] - mean[c]) / std[c];

    cv::Mat normalized;
    cv::merge(channels, normalized);
    return normalized;
}

cv::Mat resizeImage(const cv::Mat& image, int height, int width)
{
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(width, height),
               0, 0, cv::INTER_LINEAR);
    return resized;
}

at::Tensor cvToTensor(const cv::Mat& image)
{
    // image is CV_32FC3, HxWxC
    cv::Mat cont;
    if (!image.isContinuous())
        image.copyTo(cont);
    else
        cont = image;

    // Wrap as tensor (H, W, C)
    auto tensor = torch::from_blob(
        cont.data,
        {1, cont.rows, cont.cols, cont.channels()},
        torch::kFloat32
    ).clone();

    // (1, H, W, C) -> (1, C, H, W)
    tensor = tensor.permute({0, 3, 1, 2});
    return tensor;
}

cv::Mat tensorToCv(const at::Tensor& tensor)
{
    // tensor: (H, W) float32
    at::Tensor cpu = tensor.squeeze().to(torch::kCPU).contiguous();
    cv::Mat mat(cpu.size(0), cpu.size(1), CV_32FC1,
                cpu.data_ptr<float>());
    return mat.clone();
}

}  // namespace Sam3
