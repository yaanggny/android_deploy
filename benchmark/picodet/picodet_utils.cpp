#include "picodet_utils.h"

#include <opencv2/imgproc.hpp>

namespace picodet {
void loadModel(const char *modelPrefix, int input_width, bool useGpu, PicoDet &detector)
{
    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modelPrefix);
    sprintf(modelpath, "%s.bin", modelPrefix);
    
    // detector = PicoDet(parampath, modelpath, input_width, input_width, useGpu, 0.45, 0.3);
    detector.load(parampath, modelpath, input_width, input_width, useGpu, 0.45, 0.3);
}

static std::vector<int> GenerateColorMap(int num_class) {
  auto colormap = std::vector<int>(3 * num_class, 0);
  for (int i = 0; i < num_class; ++i) {
    int j = 0;
    int lab = i;
    while (lab) {
      colormap[i * 3] |= (((lab >> 0) & 1) << (7 - j));
      colormap[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j));
      colormap[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j));
      ++j;
      lab >>= 3;
    }
  }
  return colormap;
}

cv::Mat draw_bboxes(const cv::Mat &im, const std::vector<BoxInfo> &bboxes) 
{
  static const char *class_names[] = {
      "person",        "bicycle",      "car",
      "motorcycle",    "airplane",     "bus",
      "train",         "truck",        "boat",
      "traffic light", "fire hydrant", "stop sign",
      "parking meter", "bench",        "bird",
      "cat",           "dog",          "horse",
      "sheep",         "cow",          "elephant",
      "bear",          "zebra",        "giraffe",
      "backpack",      "umbrella",     "handbag",
      "tie",           "suitcase",     "frisbee",
      "skis",          "snowboard",    "sports ball",
      "kite",          "baseball bat", "baseball glove",
      "skateboard",    "surfboard",    "tennis racket",
      "bottle",        "wine glass",   "cup",
      "fork",          "knife",        "spoon",
      "bowl",          "banana",       "apple",
      "sandwich",      "orange",       "broccoli",
      "carrot",        "hot dog",      "pizza",
      "donut",         "cake",         "chair",
      "couch",         "potted plant", "bed",
      "dining table",  "toilet",       "tv",
      "laptop",        "mouse",        "remote",
      "keyboard",      "cell phone",   "microwave",
      "oven",          "toaster",      "sink",
      "refrigerator",  "book",         "clock",
      "vase",          "scissors",     "teddy bear",
      "hair drier",    "toothbrush"};

  cv::Mat image = im.clone();
  int src_w = image.cols;
  int src_h = image.rows;
  int thickness = 2;
  auto colormap = GenerateColorMap(sizeof(class_names));

  for (size_t i = 0; i < bboxes.size(); i++) {
    const BoxInfo &bbox = bboxes[i];
    // std::cout << bbox.x1 << ". " << bbox.y1 << ". " << bbox.x2 << ". "
    //           << bbox.y2 << ". " << std::endl;
    int c1 = colormap[3 * bbox.label + 0];
    int c2 = colormap[3 * bbox.label + 1];
    int c3 = colormap[3 * bbox.label + 2];
    cv::Scalar color = cv::Scalar(c1, c2, c3);
    // cv::Scalar color = cv::Scalar(0, 0, 255);
    cv::rectangle(image, cv::Rect(cv::Point(bbox.x1, bbox.y1),
                                  cv::Point(bbox.x2, bbox.y2)),
                  color, 1);

    char text[256];
    sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.score * 100);

    int baseLine = 0;
    cv::Size label_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

    int x = bbox.x1;
    int y = bbox.y1 - label_size.height - baseLine;
    if (y < 0)
      y = 0;
    if (x + label_size.width > image.cols)
      x = image.cols - label_size.width;

    cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                  cv::Size(label_size.width,
                                           label_size.height + baseLine)),
                  color, -1);

    cv::putText(image, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
  }

  return image;
}

}