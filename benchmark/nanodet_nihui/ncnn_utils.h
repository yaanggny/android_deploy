#ifndef NCNN_UTILS_H
#define NCNN_UTILS_H

#include "nanodet_.h"

void loadModel(const char* model_dir, int modelid, int cpugpu, NanoDet& detector);

void test_bench_nanodet(const cv::Mat& bgr, int nloops, int use_gpu);

#endif /* NCNN_UTILS_H */
