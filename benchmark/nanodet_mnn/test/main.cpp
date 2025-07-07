#include "test_nanodet_mnn.h"
#include <opencv2/highgui.hpp>

/*
int image_demo(const char* imagepath)
{
    nanodet_mnn::NanoDet detector = NanoDet("/data/local/tmp/ncnn/weights/nanodet-plus-m_416_mnn.mnn");
    
    cv::Mat image = cv::imread(imagepath);
    
    int height = detector.input_size[0];
    int width = detector.input_size[1];
    
    nanodet_mnn::object_rect effect_roi;
    cv::Mat resized_img;
    nanodet_mnn::resize_uniform(image, resized_img, cv::Size(width, height), effect_roi);
    std::vector<nanodet_mnn::BoxInfo> results;
    detector.detect(resized_img, results);
    
    nanodet_mnn::draw_bboxes(image, results, effect_roi, "result_mnn.jpg");
    return 0;
}
*/

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "usage: %s <imgFile>", argv[0]);
        return -1;
    }
    // NanoDet detector = NanoDet("../model/nanodet-160.mnn", 160, 160, 4, 0.4, 0.3);
    nanodet_mnn::image_demo(argv[1]);
}
