// processor_mixins.hpp
// Author: Jason Hughes
// Date:   2026
//
// Low-level image manipulation helpers shared by the processor.

#pragma once

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

namespace Sam3
{

/// Convert BGR OpenCV image to float32 RGB, normalize per-channel, and
/// return as an OpenCV Mat of type CV_32FC3.
///
/// @param image   Input BGR uint8 image
/// @param mean    Per-channel mean  {R, G, B}
/// @param std     Per-channel std   {R, G, B}
cv::Mat normalizeImage(const cv::Mat& image,
                       const std::vector<float>& mean,
                       const std::vector<float>& std);

/// Resize an image to (width, height) using bilinear interpolation.
cv::Mat resizeImage(const cv::Mat& image, int height, int width);

/// Convert a normalized HxWxC float32 OpenCV Mat to a (C, H, W) torch Tensor.
at::Tensor cvToTensor(const cv::Mat& image);

/// Convert a single-channel (H, W) torch float Tensor to a CV_32FC1 Mat.
cv::Mat tensorToCv(const at::Tensor& tensor);

}  // namespace Sam3
