#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "timer.h"
#include "YoloV5.h"
#include "yolo11.h"

// #define USE_NANODET_ORI

#ifdef USE_NANODET_ORI
#include "NanoDet.h"
#else
#include "nanodet/ncnn_utils.h"
#endif

#include <string>

static std::vector<std::string> labels{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};

static void drawBboxes(cv::Mat& img, const std::vector<BoxInfo>& bboxes)
{
    static const unsigned char colors[19][3] = {
            {54,  67,  244},
            {99,  30,  233},
            {176, 39,  156},
            {183, 58,  103},
            {181, 81,  63},
            {243, 150, 33},
            {244, 169, 3},
            {212, 188, 0},
            {136, 150, 0},
            {80,  175, 76},
            {74,  195, 139},
            {57,  220, 205},
            {59,  235, 255},
            {7,   193, 255},
            {0,   152, 255},
            {34,  87,  255},
            {72,  85,  121},
            {158, 158, 158},
            {139, 125, 96}
    };


    for (size_t i = 0; i < bboxes.size(); i++) 
    {
        const auto& b = bboxes[i];

        const unsigned char *color = colors[i % 19];
        cv::Scalar cc(color[0], color[1], color[2]);

        cv::Rect rect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);

        cv::rectangle(img, rect, cc, 2);

        char text[256];
        sprintf(text, "%s %.2f", labels[b.label].c_str(), b.score);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = rect.x;
        int y = rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > img.cols)
            x = img.cols - label_size.width;
        cv::rectangle(img, cv::Rect(cv::Point(x, y),
                                    cv::Size(label_size.width, label_size.height + baseLine)), cc,
                    -1);
        cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0)
                                                                    : cv::Scalar(255, 255, 255);
        cv::putText(img, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    textcc, 1);
    }
}

void saveImg(const std::string& fname, const cv::Mat& img)
{
    cv::imwrite(fname, img);
    printf("save: %s\n\n", fname.c_str());
}

int main(const int argc, const char* argv[])
{
    if(argc < 4) 
    {
        printf("Usage: %s <yolov5model_prefix> <nanoDet_prefix> <image_path> <use_gpu{0,1}>\n", argv[0]);
        return -1;
    }
    std::string model_prefix(argv[1]);
    std::string modelnano_prefix(argv[2]);
    std::string image_path(argv[3]);
    int use_gpu = 0;
    if (argc > 4)
        use_gpu = atoi(argv[4]);

    cv::Mat bgr = cv::imread(image_path, 1);

    printf("img: %s  shape: %d x %d\n", image_path.c_str(), bgr.cols, bgr.rows);

    cv::Mat bgr_cpy = bgr.clone();

    // draw full-width text with myfont
    BenchmarkTimer tm("Detector", 5, 1);
    tm.start();

    std::string fo;
    {
        YoloV5::detector = new YoloV5(model_prefix, true);
        tm.getDt("yolov5_init");
        std::vector<BoxInfo> bboxes = YoloV5::detector->detect(bgr, 0.3, 0.4);
        printf("bboxes size: %d\n", (int)bboxes.size());
        tm.getDt("detect");
        drawBboxes(bgr_cpy, bboxes);
        fo = "result_yolov5.jpg";
        saveImg(fo, bgr_cpy);
    }

    {
        YOLO11_det detector;
        const char* model_prefix = "/data/local/tmp/ncnn/weights/yolo11n.ncnn";
        tm.start();
        detector.load(model_prefix, use_gpu);
        detector.set_det_target_size(320);
        tm.getDt("yolov11n_init");
        std::vector<Object> bboxes2;
        detector.detect(bgr, bboxes2);
        printf("bboxes size: %d\n", (int)bboxes2.size());
        tm.getDt("yolov11n-detect");
        bgr_cpy = bgr.clone();
        detector.draw(bgr_cpy, bboxes2);
        fo = "result_yolov11n.jpg";
        saveImg(fo, bgr_cpy);
    }

    {
        YOLOv8_det_coco detector;
        // const char* model_prefix = "/data/local/tmp/ncnn/weights/yolov8n.ncnn";
        const char* model_prefix = "/data/local/tmp/ncnn/weights/best_pnnx.py.ncnn";
        tm.start();
        detector.load(model_prefix, use_gpu);
        detector.set_det_target_size(320);
        tm.getDt("yolov8n_init");
        std::vector<Object> bboxes2;
        detector.detect(bgr, bboxes2);
        printf("bboxes size: %d\n", (int)bboxes2.size());
        tm.getDt("yolov8n-detect");
        bgr_cpy = bgr.clone();
        detector.draw(bgr_cpy, bboxes2);
        fo = "result_yolov8n.jpg";
        saveImg(fo, bgr_cpy);
    }
    
    // fp16
    // {
    //     YOLOv8_det_coco detector;
    //     // const char* model_prefix = "/data/local/tmp/ncnn/weights/yolov8n_pnnx.py.ncnn-fp16";
    //     // const char* model_prefix = "/data/local/tmp/ncnn/weights/yolov8n_pnnx.py.ncnn-opt-f16";
    //     const char* model_prefix = "/data/local/tmp/ncnn/weights/yolov8n_pnnx.py.ncnn-f16";
    //     tm.start();
    //     detector.load(model_prefix, use_gpu);
    //     detector.set_det_target_size(320);
    //     tm.getDt("fp16-yolov8n_init");
    //     std::vector<Object> bboxes2;
    //     detector.detect(bgr, bboxes2);
    //     printf("bboxes size: %d\n", (int)bboxes2.size());
    //     tm.getDt("fp16-yolov8n-detect");
    //     bgr_cpy = bgr.clone();
    //     detector.draw(bgr_cpy, bboxes2);
    //     fo = "result_yolov8n_fp16.jpg";
    //     saveImg(fo, bgr_cpy);
    // }

    // int8
    {
        YOLOv8_det_coco detector;
        const char* model_prefix = "/data/local/tmp/ncnn/weights/yolov8n_pnnx.py.ncnn-f32int8";
        tm.start();
        detector.load(model_prefix, use_gpu);
        detector.set_det_target_size(320);
        tm.getDt("int8-yolov8n_init");
        std::vector<Object> bboxes2;
        detector.detect(bgr, bboxes2);
        printf("bboxes size: %d\n", (int)bboxes2.size());
        tm.getDt("int8-yolov8n-detect");
        bgr_cpy = bgr.clone();
        detector.draw(bgr_cpy, bboxes2);
        fo = "result_yolov8n_int8.jpg";
        saveImg(fo, bgr_cpy);
    }

    
#ifdef USE_NANODET_ORI
    {
        tm.start();
        NanoDet::detector = new NanoDet(modelnano_prefix, true);
        tm.getDt("nano_init");
        std::vector<BoxInfo> bboxes2 = NanoDet::detector->detect(bgr, 0.3, 0.4);
        printf("bboxes size: %d\n", (int)bboxes2.size());
        tm.getDt("detect");

        bgr_cpy = bgr.clone();
        drawBboxes(bgr_cpy, bboxes2);
        fo = "result_nano-ori.jpg";
        saveImg(fo, bgr_cpy);
    }
#else
    {
        NanoDet detector;
        const char* model_dir = "/data/local/tmp/ncnn/weights";
        tm.start();
        loadModel(model_dir, 0, use_gpu, detector);
        tm.getDt("nano_init");
        {
            std::vector<ObjectNano> bboxes2;
            detector.detect(bgr, bboxes2);
            tm.getDt("nano_detect");
            bgr_cpy = bgr.clone();
            detector.draw(bgr_cpy, bboxes2);
        }
        fo = "result_nano_nihui.jpg";
        saveImg(fo, bgr_cpy);
    }

    // int8
    {
        NanoDet detector;
        const char* model_dir = "/data/local/tmp/ncnn/weights";
        tm.start();
        loadModel(model_dir, 2, use_gpu, detector);
        tm.getDt("nano_init");
        {
            std::vector<ObjectNano> bboxes2;
            detector.detect(bgr, bboxes2);
            tm.getDt("nano_detect");
            bgr_cpy = bgr.clone();
            detector.draw(bgr_cpy, bboxes2);
        }
        fo = "result_nano_nihui-int8.jpg";
        saveImg(fo, bgr_cpy);
    }
#endif

    return 0;
}