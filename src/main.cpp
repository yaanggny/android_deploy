#include "yolov5.h"
#include "detect_utils.h"

#include <cstdio>
#include <opencv2/highgui.hpp>
// #include <opencv2/core/utility.hpp>

#include <iostream>
#include <string>

int main(const int argc, const char** argv) 
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <model_name> <img_path>" << std::endl;
        return 0;
    }

    YoloV5 net(argv[1]);
    std::string img_dir(argv[2]);
    // std::vector<std::string> imgFiles;
    std::vector<cv::String> imgFiles;
    if (img_dir.find(".jpeg") != std::string::npos)
    {
        imgFiles.push_back(img_dir);
    }
    else
    {
        cv::glob(img_dir + "/*.jpeg", imgFiles, false);
    }

    int cnt = 0;
    for (auto& imgFile : imgFiles)
    {
        printf("Detecting %s\n", imgFile.c_str());
        cv::Mat img = cv::imread(imgFile, 1);
        // std::vector<cv::Rect> boxes;
        std::vector<Object> boxes;
        net.detect(img, boxes);

        cv::Mat img_boxes = img.clone();
        net.draw_bboxes(img_boxes, boxes);

        cv::imwrite("detected_" + std::to_string(cnt) + ".jpg", img_boxes);
        cnt++;
    }    

    return 0;
}