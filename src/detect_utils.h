#ifndef DETECT_UTILS_H
#define DETECT_UTILS_H

#include <vector>
#include <opencv2/core.hpp>

#include <net.h>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

void qsort_descent_inplace(std::vector<Object> &objects);

void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked,
                              float nms_threshold);

void generate_proposals(const ncnn::Mat &anchors, int stride, const ncnn::Mat &in_pad,
                               const ncnn::Mat &feat_blob, float prob_threshold,
                               std::vector<Object> &objects);

#endif /* DETECT_UTILS_H */
