#ifndef NCNN_UTILS_H
#define NCNN_UTILS_H

#include "nanodet.h"

void loadModel(const char* model_dir, int modelid, int cpugpu, NanoDet& detector);

#endif /* NCNN_UTILS_H */
