#ifndef __YOLOV5_H
#define __YOLOV5_H

#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

#include <net.h>
#include <cpu.h>

#include <string>
#include <vector>


#include "detect_utils.h"


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

class YoloV5
{
public:
    YoloV5(const std::string& model_path_noext, int _model_size=640, bool use_gpu = false);

    void detect(const cv::Mat& img_rgb, std::vector<Object>& objects)
    {
        imgH = img_rgb.rows;
        imgW = img_rgb.cols;

        printf("input img: %d x %d\n", imgH, imgW);
        printf("model size: %d x %d\n", inH, inW);

        const int max_stride = 64;

        cv::Mat imgPreproc;
        int hpad, wpad;
        float scale = 1.f;
        preproc(img_rgb, imgPreproc, hpad, wpad);
        printf("preprocessed img: %d x %d\n", imgPreproc.rows, imgPreproc.cols);
        if (imgW > imgH)
        {
            scale = (float)inW / imgW;
        }
        else
        {
            scale = (float)inH / imgH;
        }
        ncnn::Mat imgIn = ncnn::Mat::from_pixels_resize(imgPreproc.data, ncnn::Mat::PIXEL_RGB, inW, inH, inW, inH);
        imgIn.substract_mean_normalize(meanVals, stdVals);

        printf("model input img: %d x %d x %d\n", imgIn.h, imgIn.w, imgIn.c);

        ncnn::Extractor ex = net.create_extractor();
        // ex.input("in0", imgIn);
        ex.input("images", imgIn);

        std::vector<Object> proposals;

        float prob_threshold = 0.3f;
        float nms_threshold = 0.45f;

        // stride 8
        {
            ncnn::Mat out;
            ex.extract("output", out);

            ncnn::Mat anchors(6);
            anchors[0] = 10.f;
            anchors[1] = 13.f;
            anchors[2] = 16.f;
            anchors[3] = 30.f;
            anchors[4] = 33.f;
            anchors[5] = 23.f;

            std::vector<Object> objects8;
            generate_proposals(anchors, 8, imgIn, out, prob_threshold, objects8);

            proposals.insert(proposals.end(), objects8.begin(), objects8.end());
        }

        // stride 16
        {
            ncnn::Mat out;
            ex.extract("781", out);

            ncnn::Mat anchors(6);
            anchors[0] = 30.f;
            anchors[1] = 61.f;
            anchors[2] = 62.f;
            anchors[3] = 45.f;
            anchors[4] = 59.f;
            anchors[5] = 119.f;

            std::vector<Object> objects16;
            generate_proposals(anchors, 16, imgIn, out, prob_threshold, objects16);

            proposals.insert(proposals.end(), objects16.begin(), objects16.end());
        }

        // stride 32
        {
            ncnn::Mat out;
            ex.extract("801", out);

            ncnn::Mat anchors(6);
            anchors[0] = 116.f;
            anchors[1] = 90.f;
            anchors[2] = 156.f;
            anchors[3] = 198.f;
            anchors[4] = 373.f;
            anchors[5] = 326.f;

            std::vector<Object> objects32;
            generate_proposals(anchors, 32, imgIn, out, prob_threshold, objects32);

            proposals.insert(proposals.end(), objects32.begin(), objects32.end());
        }

        // sort all proposals by score from highest to lowest
        qsort_descent_inplace(proposals);

        // apply nms with nms_threshold
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, nms_threshold);

        int count = picked.size();
        printf("detected_object num = %d\n", count);

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].x - (wpad / 2)) / scale;
            float y0 = (objects[i].y - (hpad / 2)) / scale;
            float x1 = (objects[i].x + objects[i].w - (wpad / 2)) / scale;
            float y1 = (objects[i].y + objects[i].h - (hpad / 2)) / scale;

            // clip
            x0 = std::max(std::min(x0, (float)(imgW - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(imgH - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(imgW - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(imgH - 1)), 0.f);

            objects[i].x = x0;
            objects[i].y = y0;
            objects[i].w = x1 - x0;
            objects[i].h = y1 - y0;
        }
    }

    void preproc(const cv::Mat& img, cv::Mat& out, int& ph, int& pw);

    void draw_bboxes(cv::Mat& img, const std::vector<Object>& bbox)
    {
        color_index = 0;
        for (const auto& obj : bbox)
        {
            cv::Rect rect(obj.x, obj.y, obj.w, obj.h);
            draw_bbox(img, rect, obj.label, obj.prob);
            color_index++;
        }
    }

    void draw_bbox(cv::Mat& img, const cv::Rect& bbox, int label, float prob)
    {
        static const char *class_names[] = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
            "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
            "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
            "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear",
            "hair drier", "toothbrush"
        };

        static const unsigned char colors[19][3] = {
            {54,  67,  244},
            {99,  30,  233},
            {176, 39,  156},
            {183, 58,  103},
            {181, 81,  63},
            {243, 150, 33},
            {244, 169, 3},
            {212, 188, 0},
            {136, 150, 0},
            {80,  175, 76},
            {74,  195, 139},
            {57,  220, 205},
            {59,  235, 255},
            {7,   193, 255},
            {0,   152, 255},
            {34,  87,  255},
            {72,  85,  121},
            {158, 158, 158},
            {139, 125, 96}
        };        
        const unsigned char *color = colors[color_index % 19];

        cv::Scalar cc(color[0], color[1], color[2]);

        const cv::Rect& b = bbox;
        cv::rectangle(img, cv::Point(b.x, b.y), cv::Point(b.x + b.width, b.y + b.height), cv::Scalar(0, 0, 255));

        char text[256];
        sprintf(text, "%s %.2f", class_names[label], prob);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = b.x;
        int y = b.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > img.cols)
            x = img.cols - label_size.width;
        cv::rectangle(img, cv::Rect(cv::Point(x, y),
                                    cv::Size(label_size.width, label_size.height + baseLine)), cc,
                      -1);
        cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0)
                                                                    : cv::Scalar(255, 255, 255);
        cv::putText(img, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    textcc, 1);
    }

private:
    ncnn::Net net;
    float meanVals[3];
    float stdVals[3];

    int imgH, imgW;
    int inH, inW;
    int model_size;

    int color_index = 0;

    // ncnn::UnlockedPoolAllocator blob_pool_allocator;
    // ncnn::PoolAllocator workspace_pool_allocator;
};


#endif /* YOLOVX_H */
