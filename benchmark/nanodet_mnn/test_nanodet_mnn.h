#ifndef TEST_NANODET_MNN_H
#define TEST_NANODET_MNN_H

#include <opencv2/core.hpp>
#include "model_base.h"

namespace nanodet_mnn {

void test_nanodet(const ModelConfig& config, const cv::Mat& bgr, int nloops, int use_gpu);
int image_demo(const char* imagepath);

}

#endif /* TEST_NANODET_MNN_H */
