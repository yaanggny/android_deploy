#ifndef PICODET_UTILS_H
#define PICODET_UTILS_H

#include "picodet.h"

namespace picodet {

void loadModel(const char *model_prefix, int input_width, bool useGpu, PicoDet &detector);

cv::Mat draw_bboxes(const cv::Mat &im, const std::vector<BoxInfo> &bboxes);
}

#endif /* PICODET_UTILS_H */
