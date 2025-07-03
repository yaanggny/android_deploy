#include "NanoDet.h"
#include "YoloV5.h"

#include "timer.h"
#include "io_img.h"

#include <opencv2/imgproc.hpp>
#include <fmt/format.h>

#include <string>

namespace nanodet {

static std::vector<std::string> labels{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};

static void drawBboxes(cv::Mat& img, const std::vector<nanodet::BoxInfo>& bboxes)
{
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


    for (size_t i = 0; i < bboxes.size(); i++) 
    {
        const auto& b = bboxes[i];

        const unsigned char *color = colors[i % 19];
        cv::Scalar cc(color[0], color[1], color[2]);

        cv::Rect rect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);

        cv::rectangle(img, rect, cc, 2);

        char text[256];
        sprintf(text, "%s %.2f", labels[b.label].c_str(), b.score);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = rect.x;
        int y = rect.y - label_size.height - baseLine;
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
}


void test_nanodet(const ModelConfig& config, const cv::Mat& bgr, int nloops, int use_gpu)
{
    BenchmarkTimer tm("NanoDet-ori", 5, 1);
    const std::string modelDir = "/data/local/tmp/ncnn/weights/";
    // const std::string modelName = "nanodet-plus-m_416";
    const std::string modelName = config.modelname_noext;
    const std::string modelPrefix = modelDir + modelName;

    NanoDet detector(modelPrefix, true);
    detector.setModelConfig(config);
    tm.getDt("nanodet_init");

    cv::Mat bgr_cpy;
    for (int i = 0; i < nloops; i++)
    {
        tm.start();
        std::vector<BoxInfo> bboxes = detector.detect(bgr, 0.3, 0.4);
        tm.stop();

        bgr_cpy = bgr.clone();
        drawBboxes(bgr_cpy, bboxes);
        std::string fo = fmt::format("result_{}_{}.jpg", modelName, i);
        saveImg(fo, bgr_cpy);
    }
}

void test_yolov5s(const cv::Mat& bgr, int nloops, int use_gpu)
{
    BenchmarkTimer tm("YoloV5s-f32", 5, 1);
    const std::string modelDir = "/data/local/tmp/ncnn/weights/";
    const std::string modelName = "yolov5s";
    const std::string modelPrefix = modelDir + modelName;
    YoloV5::detector = new YoloV5(modelPrefix, true);
    tm.getDt("yolov5_init");

    cv::Mat bgr_cpy;
    for (int i = 0; i < nloops; i++)
    {
        tm.start();
        std::vector<nanodet::BoxInfo> bboxes = YoloV5::detector->detect(bgr, 0.3, 0.4);
        tm.stop();

        bgr_cpy = bgr.clone();
        drawBboxes(bgr_cpy, bboxes);
        std::string fo = fmt::format("result_{}_{}.jpg", modelName, i);
        saveImg(fo, bgr_cpy);
    }
}

}