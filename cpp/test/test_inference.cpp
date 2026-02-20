// test_inference.cpp
// Author: Jason Hughes
// Date:   2026
//
// End-to-end SAM3 inference test.
// Loads test.png, segments the "road" concept, and displays the result.
//
// Usage:
//   ./sam3_inference_test

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "sam3/sam3_model.hpp"
#include "sam3/processor.hpp"

static const std::string MODELS_DIR = "../models";
static const std::string CONFIG_DIR = "../config";
static const std::string IMAGE_PATH = "../../test.png";
static const std::string TEXT_PROMPT = "road";

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);

    // -----------------------------------------------------------------------
    // Load model and processor
    // -----------------------------------------------------------------------
    std::cout << "Loading SAM3 models from: " << MODELS_DIR << "\n";
    Sam3::Sam3 model(MODELS_DIR);

    Sam3::Sam3Processor processor(
        CONFIG_DIR + "/sam3.yaml",
        CONFIG_DIR + "/merges.txt",
        CONFIG_DIR + "/vocab.json"
    );

    // -----------------------------------------------------------------------
    // Load image
    // -----------------------------------------------------------------------
    cv::Mat image = cv::imread(IMAGE_PATH);
    if (image.empty())
    {
        std::cerr << "Error: could not read image: " << IMAGE_PATH << "\n";
        return 1;
    }
    std::cout << "Image: " << IMAGE_PATH
              << "  (" << image.cols << "x" << image.rows << ")\n";
    std::cout << "Prompt: \"" << TEXT_PROMPT << "\"\n";

    // -----------------------------------------------------------------------
    // Pre-process
    // -----------------------------------------------------------------------
    Sam3::Sam3ModelInputs inputs = processor.process(image, TEXT_PROMPT);

    // -----------------------------------------------------------------------
    // Inference
    // -----------------------------------------------------------------------
    std::cout << "Running inference ...\n";

    // Cache image features (useful if querying multiple text prompts)
    model.setImage(inputs.pixel_values, inputs.orig_height, inputs.orig_width);

    // Cache text features
    model.setText(inputs.input_ids, inputs.attention_mask);

    // Decode masks
    Sam3::Sam3Output output = model.inference();

    std::cout << "pred_masks:      " << output.pred_masks.sizes()      << "\n";
    std::cout << "pred_logits:     " << output.pred_logits.sizes()     << "\n";
    std::cout << "presence_logits: " << output.presence_logits.sizes() << "\n";

    // -----------------------------------------------------------------------
    // Post-process
    // -----------------------------------------------------------------------
    std::vector<cv::Mat> masks = processor.postProcess(
        output.pred_masks,
        output.pred_logits,
        output.presence_logits,
        inputs.orig_height,
        inputs.orig_width
    );
    std::cout << "Instances found: " << masks.size() << "\n";

    // -----------------------------------------------------------------------
    // Visualize: colour overlay per instance
    // -----------------------------------------------------------------------
    cv::Mat overlay;
    image.copyTo(overlay);

    const std::vector<cv::Vec3b> colours = {
        {0,200,0}, {220,0,0}, {0,0,220}, {0,200,200}, {200,0,200}, {200,200,0}
    };
    for (size_t i = 0; i < masks.size(); ++i)
    {
        cv::Vec3b col = colours[i % colours.size()];
        for (int r = 0; r < overlay.rows; ++r)
            for (int c = 0; c < overlay.cols; ++c)
                if (masks[i].at<uchar>(r, c) > 0)
                    overlay.at<cv::Vec3b>(r, c) =
                        0.55 * col + 0.45 * image.at<cv::Vec3b>(r, c);
    }

    cv::Mat mask;
    cv::Mat combined = cv::Mat::zeros(image.size(), CV_8UC1);
    for (const auto& m : masks)
        cv::bitwise_or(combined, m, combined);
    mask = combined;

    // Save result
    std::string out_path = "sam3_result.png";
    cv::imwrite(out_path, overlay);
    std::cout << "Saved result to: " << out_path << "\n";

    // Display (optional – comment out if running headless)
    cv::imshow("SAM3 segmentation: " + TEXT_PROMPT, overlay);
    cv::imshow("Mask", mask);
    std::cout << "Press any key to exit.\n";
    cv::waitKey(0);

    return 0;
}
