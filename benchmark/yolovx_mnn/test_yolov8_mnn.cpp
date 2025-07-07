#include "yolov8_mnn.h"
#include <opencv2/highgui.hpp>

#include "timer.h"
#include "io_img.h"

namespace yolov8_mnn {

void test_yolovx_mnn(const ModelConfig& config, const cv::Mat& bgr, int nloops, int use_gpu)
{
    BenchmarkTimer tm(config.modelname_noext.c_str(), 5, 1);
    const std::string modelDir = "/data/local/tmp/ncnn/weights/";
    // const std::string modelName = "nanodet-plus-m_416";
    const std::string modelName = config.modelname_noext;
    const std::string modelPrefix = modelDir + modelName;

    YolovxDet detector(modelPrefix + ".mnn", 4);

    detector.setModelConfig(config);
    tm.getDt("init");

    cv::Mat bgr_cpy;
    for (int i = 0; i < nloops; i++)
    {
        tm.start();
        std::vector<BoxInfo> bboxes;
        detector.detect(bgr, bboxes);
        tm.stop();

        fmt::print("nboxes: {}\n", bboxes.size());

        bgr_cpy = bgr.clone();
        detector.drawBoxes(bgr_cpy, bboxes);
        std::string fo = fmt::format("result_{}_{}.jpg", modelName+"_mnn", i);
        saveImg(fo, bgr_cpy);
    }
}


int image_demo(const char* imagepath)
{
    ModelConfig cfg;
    cfg.modelname_noext = "yolov8n_320_ts_mnn";
    cfg.inputSize = 416;
    cfg.strides = {8, 16, 32};
    cfg.inputNode = "data";
    cfg.outputNode = "output";

    cv::Mat image = cv::imread(imagepath);

    test_yolovx_mnn(cfg, image, 1, 0);

    return 0;
}

}
