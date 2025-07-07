#ifndef YOLOV8_MNN_H
#define YOLOV8_MNN_H

#pragma once

#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"

#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "model_base.h"

namespace yolov8_mnn
{

struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
    BoxInfo(float _x1, float _y1, float _x2, float _y2, float _score, int _label)
    : x1(_x1), y1(_y1), x2(_x2), y2(_y2), score(_score), label(_label) {}
};


class YolovxDet 
{
public:
    YolovxDet(const std::string& mnn_path, int num_thread_ = 4);

    ~YolovxDet();

    int detect(const cv::Mat& img, std::vector<BoxInfo>& boxes, float score_threshold = 0.4, float nms_threshold = 0.45);
    int drawBoxes(cv::Mat& rgb, const std::vector<BoxInfo>& boxes);

    void setModelConfig(int inputSize, const std::vector<int>& strides, const std::string& inputNode, const std::string& outputNode);
    void setModelConfig(const ModelConfig& config);

    // modify these parameters to the same with your config if you want to use your own model
    int inputSize[2] = {320, 320}; // input height and width
    int numClass = 80; // number of classes. 80 for COCO
    int regMax = 7; // `reg_max` set in the training config. Default: 7.
    std::vector<int> strides = { 8, 16, 32 }; // strides of the multi-level feature.
    

    struct PreprocInfo
    {
        int imgH, imgW;
        int hrz, wrz;
        int hpad, wpad;
    };

private:
    int preproc(const cv::Mat& img, PreprocInfo& info);
    int postproc(const PreprocInfo& info, std::vector<BoxInfo>& dets);
private:

    std::shared_ptr<MNN::Interpreter> interpreter;
    MNN::Session *session = nullptr;
    MNN::Tensor *inputTensor = nullptr;

    ModelConfig m_config;

    // const float mean_vals[3] = { 103.53f, 116.28f, 123.675f };
    // const float norm_vals[3] = { 0.017429f, 0.017507f, 0.017125f };
    const float mean_vals[3] = { 0, 0, 0 };
    const float norm_vals[3] = { 1.0f/255, 1.0f/255, 1.0f/255 };
};

}

#endif /* YOLOV8_MNN_H */
