#ifndef TEST_NANODET_H
#define TEST_NANODET_H

#include <opencv2/core.hpp>
#include "NanoDet.h"

namespace nanodet {

void test_nanodet(const ModelConfig& config, const cv::Mat& bgr, int nloops, int use_gpu);
void test_yolov5s(const cv::Mat& bgr, int nloops, int use_gpu);

}


#endif /* TEST_NANODET_H */
