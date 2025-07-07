#ifndef TEST_YOLOV8_MNN_H
#define TEST_YOLOV8_MNN_H

#include <opencv2/core.hpp>
#include "model_base.h"

namespace yolov8_mnn {

void test_yolovx_mnn(const ModelConfig& config, const cv::Mat& bgr, int nloops, int use_gpu);

}

#endif /* TEST_YOLOV8_MNN_H */
