#include "io_img.h"

#include <opencv2/highgui.hpp>

void saveImg(const std::string& fname, const cv::Mat& img)
{
#ifdef _DEBUG_SAVE
    cv::imwrite(fname, img);
    printf("save: %s\n", fname.c_str());
#endif 
}