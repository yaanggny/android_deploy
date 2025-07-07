#ifndef __NanoDet_H__
#define __NanoDet_H__

#pragma once

#include "Interpreter.hpp"

#include "MNNDefine.h"
#include "Tensor.hpp"
#include "ImageProcess.hpp"

#include <opencv2/core.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

#include "model_base.h"

namespace nanodet_mnn
{


typedef struct HeadInfo_
{
    std::string cls_layer;
    std::string dis_layer;
    int stride;
} HeadInfo;

typedef struct BoxInfo_
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

typedef struct CenterPrior_
{
    int x;
    int y;
    int stride;
} CenterPrior;

class NanoDet {
public:
    NanoDet(const std::string &mnn_path, int num_thread_ = 4);

    ~NanoDet();

    int detect(cv::Mat &img, std::vector<BoxInfo> &result_list, float score_threshold_ = 0.5, float nms_threshold_ = 0.3);
    std::string get_label_str(int label);
    void setModelConfig(int inputSize, const std::vector<int>& strides, const std::string& inputNode, const std::string& outputNode);
    void setModelConfig(const ModelConfig& config);

    // modify these parameters to the same with your config if you want to use your own model
    int input_size[2] = {416, 416}; // input height and width
    int num_class = 80; // number of classes. 80 for COCO
    int reg_max = 7; // `reg_max` set in the training config. Default: 7.
    std::vector<int> strides = { 8, 16, 32, 64 }; // strides of the multi-level feature.

private:
    void decode_infer(MNN::Tensor *pred, std::vector<CenterPrior>& center_priors, float threshold, std::vector<std::vector<BoxInfo>> &results);
    BoxInfo disPred2Bbox(const float *&dfl_det, int label, float score, int x, int y, int stride);
    void nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH);

private:

    std::shared_ptr<MNN::Interpreter> NanoDet_interpreter;
    MNN::Session *NanoDet_session = nullptr;
    MNN::Tensor *input_tensor = nullptr;

    int image_w;
    int image_h;

    float score_threshold;
    float nms_threshold;

    ModelConfig m_config;

    const float mean_vals[3] = { 103.53f, 116.28f, 123.675f };
    const float norm_vals[3] = { 0.017429f, 0.017507f, 0.017125f };
};

template <typename _Tp>
int activation_function_softmax(const _Tp *src, _Tp *dst, int length);

inline float fast_exp(float x);
inline float sigmoid(float x);

    
} // namespace nanodet_mnn

#endif // __NanoDet_H__
