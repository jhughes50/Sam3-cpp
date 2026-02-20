// test_road.cpp
// Author: Jason Hughes
// Date:   2026
//
// Loads test.png (downscaled 4x), segments "road", saves mask and overlay.
// Mirrors the behaviour of python/test_sam3.py for direct comparison.

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "sam3/sam3_model.hpp"
#include "sam3/processor.hpp"

static const std::string MODELS_DIR  = "../models";
static const std::string CONFIG_DIR  = "../config";
static const std::string IMAGE_PATH  = "../../test.png";
static const std::string TEXT_PROMPT = "road";
static const int         DOWNSCALE   = 4;

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);

    std::cout << "Loading SAM3 models ...\n";
    Sam3::Sam3 model(MODELS_DIR);

    Sam3::Sam3Processor processor(CONFIG_DIR + "/sam3.yaml", CONFIG_DIR + "/merges.txt", CONFIG_DIR + "/vocab.json");

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

    std::cout << "Image: " << full.cols << "x" << full.rows << " -> " << image.cols << "x" << image.rows << "\n";
    std::cout << "Prompt: \"" << TEXT_PROMPT << "\"\n";

    Sam3::Sam3ModelInputs inputs = processor.process(image, TEXT_PROMPT);

    std::cout << "Running inference ...\n";
    model.setImage(inputs.pixel_values, inputs.orig_height, inputs.orig_width);
    model.setText(inputs.input_ids, inputs.attention_mask);
    // initial slow trace to initalize cuDNN
    Sam3::Sam3Output output = model.inference();
    
    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        model.setImage(inputs.pixel_values, inputs.orig_height, inputs.orig_width);
        output = model.inference();
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration = end - start;
        
        std::cout << "Test: " << i << " Inference took: " << duration.count() << " ms" << std::endl;
    }
    std::cout << "pred_masks:      " << output.pred_masks.sizes()      << "\n";
    std::cout << "pred_logits:     " << output.pred_logits.sizes()     << "\n";
    std::cout << "presence_logits: " << output.presence_logits.sizes() << "\n";

    std::vector<cv::Mat> masks = processor.postProcess(
        output.pred_masks,
        output.pred_logits,
        output.presence_logits,
        inputs.orig_height,
        inputs.orig_width
    );
    std::cout << "Instances found: " << masks.size() << "\n";

    const std::vector<cv::Vec3b> colours = {
        {0,200,0}, {220,0,0}, {0,0,220},
        {0,200,200}, {200,0,200}, {200,200,0}
    };

    cv::Mat overlay = image.clone();
    cv::Mat combined_mask = cv::Mat::zeros(image.size(), CV_8UC1);

    for (size_t i = 0; i < masks.size(); ++i)
    {
        cv::Vec3b col = colours[i % colours.size()];
        for (int r = 0; r < overlay.rows; ++r)
            for (int c = 0; c < overlay.cols; ++c)
                if (masks[i].at<uchar>(r, c) > 0)
                {
                    cv::Vec3b pix = image.at<cv::Vec3b>(r, c);
                    overlay.at<cv::Vec3b>(r, c) = cv::Vec3b(
                        static_cast<uchar>(0.45f * pix[0] + 0.55f * col[0]),
                        static_cast<uchar>(0.45f * pix[1] + 0.55f * col[1]),
                        static_cast<uchar>(0.45f * pix[2] + 0.55f * col[2])
                    );
                    combined_mask.at<uchar>(r, c) = 255;
                }
    }

    const std::string overlay_path = "../../cpp_result.png";
    const std::string mask_path    = "../../cpp_mask.png";

    cv::imwrite(overlay_path,  overlay);
    cv::imwrite(mask_path,     combined_mask);

    std::cout << "Saved: " << overlay_path << "\n";
    std::cout << "Saved: " << mask_path    << "\n";

    return 0;
}
