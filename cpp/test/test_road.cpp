// test_road.cpp
// Author: Jason Hughes
// Date:   2026
//
// Loads test.png (downscaled 4x), segments multiple text classes ("road", "car").
// Image is encoded once; text encoder + mask decoder run once per class.

#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "sam3/sam3_model.hpp"
#include "sam3/processor.hpp"

static const std::string MODELS_DIR  = "../models";
static const std::string CONFIG_DIR  = "../config";
static const std::string IMAGE_PATH  = "../../test.png";
static const std::vector<std::string> TEXT_PROMPTS = {"road", "car"};
static const int         DOWNSCALE   = 4;

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);

    std::cout << "Loading SAM3 models ...\n";
    Sam3::Sam3 model(MODELS_DIR);

    Sam3::Sam3Processor processor(CONFIG_DIR + "/sam3.yaml",
                                  CONFIG_DIR + "/merges.txt",
                                  CONFIG_DIR + "/vocab.json");

    cv::Mat full = cv::imread(IMAGE_PATH);
    if (full.empty())
    {
        std::cerr << "Error: could not read " << IMAGE_PATH << "\n";
        return 1;
    }

    cv::Mat image;
    cv::resize(full, image,
               cv::Size(full.cols / DOWNSCALE, full.rows / DOWNSCALE),
               0, 0, cv::INTER_AREA);

    std::cout << "Image: " << full.cols << "x" << full.rows
              << " -> " << image.cols << "x" << image.rows << "\n";

    // Pre-process image (once) and all text prompts.
    Sam3::Sam3ModelInputs image_inputs = processor.processImageOnly(image);
    std::vector<Sam3::Sam3TextInputs> text_inputs = processor.processTexts(TEXT_PROMPTS);

    std::cout << "Prompts:";
    for (const auto& p : TEXT_PROMPTS) std::cout << " \"" << p << "\"";
    std::cout << "\n";

    // Encode image once.
    std::cout << "Encoding image ...\n";
    model.setImage(image_inputs.pixel_values, image_inputs.orig_height, image_inputs.orig_width);

    // Multi-class inference: text encoder + mask decoder run once per class,
    // image features are reused.
    std::cout << "Running multi-class inference (" << TEXT_PROMPTS.size() << " classes) ...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<Sam3::Sam3Output> outputs = model.inferenceMultiClass(text_inputs);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = t1 - t0;
    std::cout << "Inference took: " << elapsed.count() << " ms\n";

    for (int i = 0; i < 10; i++) 
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        model.setImage(image_inputs.pixel_values, image_inputs.orig_height, image_inputs.orig_width);
        std::vector<Sam3::Sam3Output> outputs = model.inferenceMultiClass(text_inputs);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = t1 - t0;
        std::cout << "Inference took: " << elapsed.count() << " ms\n";
    }
    // Colour per class.
    const std::vector<cv::Vec3b> colours = {
        {0,200,0}, {220,0,0}, {0,0,220},
        {0,200,200}, {200,0,200}, {200,200,0}
    };

    cv::Mat overlay       = image.clone();
    cv::Mat combined_mask = cv::Mat::zeros(image.size(), CV_8UC1);

    for (size_t cls = 0; cls < outputs.size(); ++cls)
    {
        std::cout << "Class \"" << TEXT_PROMPTS[cls] << "\":\n";
        std::cout << "  pred_masks:      " << outputs[cls].pred_masks.sizes()      << "\n";
        std::cout << "  pred_logits:     " << outputs[cls].pred_logits.sizes()     << "\n";
        std::cout << "  presence_logits: " << outputs[cls].presence_logits.sizes() << "\n";

        std::vector<cv::Mat> masks = processor.postProcess(
            outputs[cls].pred_masks,
            outputs[cls].pred_logits,
            outputs[cls].presence_logits,
            image_inputs.orig_height,
            image_inputs.orig_width
        );
        std::cout << "  Instances found: " << masks.size() << "\n";

        cv::Vec3b col = colours[cls % colours.size()];
        for (const auto& mask : masks)
        {
            for (int r = 0; r < overlay.rows; ++r)
                for (int c = 0; c < overlay.cols; ++c)
                    if (mask.at<uchar>(r, c) > 0)
                    {
                        cv::Vec3b pix = image.at<cv::Vec3b>(r, c);
                        overlay.at<cv::Vec3b>(r, c) = cv::Vec3b(
                            static_cast<uchar>(0.45f * pix[0] + 0.55f * col[0]),
                            static_cast<uchar>(0.45f * pix[1] + 0.55f * col[1]),
                            static_cast<uchar>(0.45f * pix[2] + 0.55f * col[2])
                        );
                        combined_mask.at<uchar>(r, c) = static_cast<uchar>((cls + 1) * 80);
                    }
        }
    }

    const std::string overlay_path = "../../cpp_result.png";
    const std::string mask_path    = "../../cpp_mask.png";

    cv::imwrite(overlay_path, overlay);
    cv::imwrite(mask_path,    combined_mask);

    std::cout << "Saved: " << overlay_path << "\n";
    std::cout << "Saved: " << mask_path    << "\n";

    return 0;
}
