//
// Created by 邓昊晴 on 14/6/2020.
//

#ifndef YOLOV5_H
#define YOLOV5_H

#include <opencv2/core.hpp>

#include <net.h>

#include <string>
#include <vector>

namespace yolocv {
    typedef struct {
        int width;
        int height;
    } YoloSize;
}

typedef struct {
    std::string name;
    int stride;
    std::vector<yolocv::YoloSize> anchors;
} YoloLayerData;

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float* outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};


class YoloV5 {
public:
    YoloV5(const std::string& modelname_noext, bool useGPU);

    ~YoloV5();

    std::vector<BoxInfo> detect(const cv::Mat& img_rgb, float threshold, float nms_threshold);
private:
    static std::vector<BoxInfo>
    decode_infer(ncnn::Mat &data, int stride, const yolocv::YoloSize &frame_size, int net_size, int num_classes,
                 const std::vector<yolocv::YoloSize> &anchors, float threshold);

    static void nms(std::vector<BoxInfo> &result, float nms_threshold);

    ncnn::Net *Net;
    int input_size = 640;
    int num_class = 80;
    std::vector<YoloLayerData> layers{
            // {"394",    32, {{116, 90}, {156, 198}, {373, 326}}},
            // {"375",    16, {{30,  61}, {62,  45},  {59,  119}}},
            {"801",    32, {{116, 90}, {156, 198}, {373, 326}}},
            {"781",    16, {{30,  61}, {62,  45},  {59,  119}}},
            {"output", 8,  {{10,  13}, {16,  30},  {33,  23}}},
    };

public:
    static YoloV5 *detector;
    static bool hasGPU;
};


#endif //YOLOV5_H
