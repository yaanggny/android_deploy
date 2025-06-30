//
// Create by RangiLyu
// 2020 / 10 / 2
//

#ifndef NANODET_H
#define NANODET_H

#include <opencv2/core.hpp>

#include <net.h>
#include <YoloV5.h>

typedef struct HeadInfo_
{
    std::string cls_layer;
    std::string dis_layer;
    int stride;
} HeadInfo;

typedef struct CenterPrior_
{
    int x;
    int y;
    int stride;
} CenterPrior;


class NanoDet{
public:
    NanoDet(const std::string& modelname_noext, bool useGPU);

    ~NanoDet();

    std::vector<BoxInfo> detect(const cv::Mat& img_rgb, float score_threshold, float nms_threshold);

private:
    void preprocess(const cv::Mat& img_rgb, ncnn::Mat& in);
    void decode_infer(ncnn::Mat& feats, std::vector<CenterPrior>& center_priors, float threshold, std::vector<std::vector<BoxInfo>>& results, float width_ratio, float height_ratio);
    BoxInfo disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride, float width_ratio, float height_ratio);

    static void nms(std::vector<BoxInfo>& result, float nms_threshold);

    ncnn::Net *Net;
    // modify these parameters to the same with your config if you want to use your own model
    int input_size[2] = {416, 416}; // input height and width
    int num_class = 80; // number of classes. 80 for COCO
    int reg_max = 7; // `reg_max` set in the training config. Default: 7.
    std::vector<int> strides = { 8, 16, 32, 64 }; // strides of the multi-level feature.

public:
    static NanoDet *detector;
    static bool hasGPU;
};


#endif //NANODET_H
