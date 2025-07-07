#include "yolov8_mnn.h"
#include "timer.h"

#include <ImageProcess.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

#include <opencv2/imgproc.hpp>

#include <fmt/printf.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cmath>

using namespace MNN;

namespace yolov8_mnn
{

inline float fast_exp(float x)
{
    union
    {
        uint32_t i;
        float f;
    } v;
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

// inline float sigmoid(float x)
// {
//     return 1.0f / (1.0f + expf(-x));
// }

template<typename _Tp>
int funcSoftmax(const _Tp* src, _Tp* dst, int length)
{
    _Tp den{ 0 };

    // memcpy(dst, src, length * sizeof(_Tp));

    for (int i = 0; i < length; ++i) 
    {
        // dst[i] = fast_exp(dst[i]);
        dst[i] = fast_exp(src[i]);
        den += dst[i];
    }

    den = 1.0f / den;

    for (int i = 0; i < length; ++i) 
    {
        dst[i] *= den;
    }

    return 0;
}

static inline float intersectionArea(const BoxInfo& a, const BoxInfo& b)
{
    // cv::Rect_<float> inter = a.rect & b.rect;
    // return inter.area();
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);
    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h; 

    return inter;
}

class YolovxPostproc
{
public:
    YolovxPostproc(const MNN::Tensor* feats, const std::vector<int>& strides, int inputSize, int regMax=7, int nc=80)
    {
        m_feats = feats;
        this->strides = strides;
        inputSizeH = inputSizeW = inputSize;
        m_regMax = regMax;
        m_nClass = nc;
    }

    void computeBoxes(const MNN::Tensor* ofeats, std::vector<BoxInfo>& boxes, 
        const std::vector<int>& strides, int regMax=7, int nClass=80, float probThreshold=0.4)
    {
        boxes.clear();

        fmt::print("Tensor: CHW= {}x{}x{}\n", ofeats->channel(), ofeats->height(), ofeats->width());

        // int nPoints = ofeats->height();
        // int nChannel = ofeats->width();
        int nPoints = ofeats->channel();
        int nChannel = ofeats->height();

        int nReg = regMax + 1;
        int nRegFeatVals = 4 * nReg;

        const float* feats = ofeats->host<float>();

        float* softmaxBuf = new float[nReg];

        int pred_row_offset = 0;
        for (size_t i = 0; i < strides.size(); i++)
        {
            int stride = strides[i];

            const int feat_w = inputSizeW / stride;
            const int feat_h = inputSizeH / stride;
            const int nPixels = feat_w * feat_h;

            int offset = nPixels * nChannel;

            // m_feats: 2100x144
            /*
            for(int y = 0; y < feat_h; y++)
            {
                for(int x = 0; x < feat_w; x++)
                {
                    const float* featsRow = feats + j*nChannel;               
                    const float* featScore = featsRow + nRegFeatVals;
                    float* maxIter = std::max_element(featScore, featScore + nClass);
                    int label = maxIter - featScore;  // maxIdx
                    float score = sigmoid(*maxIter);
                    if(score <= probThreshold) continue;
                    
                    // compute softmax
                    funcSoftmax(featsRow, softmaxBuf, nRegFeatVals);
                    float pred_ltrb[4];
                    for (int k = 0; k < 4; k++)
                    {
                        float dis = 0.f;
                        float* pointFt = softmaxBuf + k * nReg;  // 4xnReg
                        for (int l = 0; l < nReg; l++)
                        {
                            dis += l * pointFt[l];
                        }
                        pred_ltrb[k] = dis * stride;
                    }

                    float cx = (x + 0.5f) * stride;
                    float cy = (y + 0.5f) * stride;
                    float x0 = cx - pred_ltrb[0];
                    float y0 = cy - pred_ltrb[1];
                    float x1 = cx + pred_ltrb[2];
                    float y1 = cy + pred_ltrb[3];
                }
            }
            */

            for(int j = 0; j < nPixels; j++)
            {
                const float* featsRow = feats + j*nChannel;               
                const float* featScore = featsRow + nRegFeatVals;
                const float* maxIter = std::max_element(featScore, featScore + nClass);
                int label = maxIter - featScore;  // maxIdx
                float score = sigmoid(*maxIter);
                if(score <= probThreshold) continue;
                
                // compute softmax
                float pred_ltrb[4];
                for (int k = 0; k < 4; k++)
                {
                    float dis = 0.f;
                    funcSoftmax(featsRow + k * nReg, softmaxBuf, nReg);
                    for (int l = 0; l < nReg; l++)
                    {
                        dis += l * softmaxBuf[l];
                    }
                    pred_ltrb[k] = dis * stride;
                }

                int y = j / feat_w;
                int x = j % feat_w;

                float cx = (x + 0.5f) * stride;
                float cy = (y + 0.5f) * stride;
                float x0 = cx - pred_ltrb[0];
                float y0 = cy - pred_ltrb[1];
                float x1 = cx + pred_ltrb[2];
                float y1 = cy + pred_ltrb[3];

                boxes.emplace_back(BoxInfo(x0, y0, x1, y1, score, label));
            }

            feats += offset;            
        }

        delete [] softmaxBuf;
        softmaxBuf = nullptr;
    }

    void computeBoxes(std::vector<BoxInfo>& boxes, float probThreshold=0.4)
    {
        computeBoxes(m_feats, boxes, strides, m_regMax, m_nClass, probThreshold);
    }

    void nms(std::vector<BoxInfo>& boxes, std::vector<int>& picked, float nms_threshold=0.45)
    {
        picked.clear();

        // sort descending by score
        std::sort(boxes.begin(), boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });

        const int n = (int)boxes.size();
        std::vector<float> areas(n);
        for (int i = 0; i < n; i++)
        {
            const auto& b = boxes[i];
            areas[i] = (b.x2 - b.x1) * (b.y2 - b.y1);
        }

        // std::vector<int> picked;
        for (int i = 0; i < n; ++i) 
        {
            const auto& a = boxes[i];
            bool keep = true;
            for (int j = 0; j < (int)picked.size(); j++)
            {
                const auto& b = boxes[picked[j]];

                // if (a.label != b.label)
                //     continue;

                // intersection over union
                float interArea = intersectionArea(a, b);
                float unionArea = areas[i] + areas[picked[j]] - interArea;
                keep = (interArea / unionArea) <= nms_threshold;
            }
            if(keep)
                picked.push_back(i);
        }
    }

private:
    int m_regMax;
    int m_nClass;
    int inputSizeH, inputSizeW;
    std::vector<int> strides;
    const MNN::Tensor* m_feats;
};


YolovxDet::YolovxDet(const std::string& mnn_path, int num_thread_)
{
    printf("YolovxDet::YolovxDet: model= %s\n", mnn_path.c_str());
    interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));
    MNN::ScheduleConfig config;
    config.numThread = num_thread_;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode) 2;
    config.backendConfig = &backendConfig;

    session = interpreter->createSession(config);

    inputTensor = interpreter->getSessionInput(session, nullptr);
}

YolovxDet::~YolovxDet()
{
    interpreter->releaseModel();
    interpreter->releaseSession(session);
}

void YolovxDet::setModelConfig(int inputSize, const std::vector<int>& strides, const std::string& inputNode, const std::string& outputNode)
{
    this->inputSize[0] = inputSize;
    this->inputSize[1] = inputSize;
    this->strides = strides;

    m_config.inputSize = inputSize;
    m_config.strides = strides;
    m_config.inputNode = inputNode;
    m_config.outputNode = outputNode;
}

void YolovxDet::setModelConfig(const ModelConfig& config)
{
    this->inputSize[0] = config.inputSize;
    this->inputSize[1] = config.inputSize;
    this->strides = config.strides;

    m_config = config;
    m_config.printSelf();
}

int YolovxDet::preproc(const cv::Mat& img, YolovxDet::PreprocInfo& info)
{
    int imgH = img.rows;
    int imgW = img.cols;

    int ho = inputSize[0];
    int wo = inputSize[1];

    if(imgH > imgW) wo = float(ho) / imgH * imgW;
    else ho = float(wo) / imgW * imgH;

    info.imgH = imgH;
    info.imgW = imgW;
    info.hrz = ho;
    info.wrz = wo;

    // resize
    cv::Mat imgRz;
    if(ho == imgH && wo == imgW)
    {
        imgRz = img;
    }
    else
    {
        cv::resize(img, imgRz, cv::Size(wo, ho));
    }

    // pad
    int maxStride = *std::max_element(strides.begin(), strides.end());
    // int wpad = (wo + maxStride - 1) / maxStride * maxStride - wo;
    // int hpad = (ho + maxStride - 1) / maxStride * maxStride - ho;

    int hpad = std::max(inputSize[0] - ho, 0);
    int wpad = std::max(inputSize[1] - wo, 0);
    info.hpad = hpad;
    info.wpad = wpad;

    cv::Mat imgPad;
    cv::copyMakeBorder(imgRz, imgPad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    interpreter->resizeTensor(inputTensor, {1, 3, inputSize[0], inputSize[1]});
    interpreter->resizeSession(session);
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(
        MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::BGR, mean_vals, 3, norm_vals, 3));
    pretreat->convert(imgPad.data, inputSize[1], inputSize[0], imgPad.step[0], inputTensor);

    fmt::print("inputTensor: BCHW: {}x{}x{}x{}  dims={}  dimType={}\n", inputTensor->batch(), inputTensor->channel(), 
        inputTensor->height(), inputTensor->width(), inputTensor->dimensions(), (int)inputTensor->getDimensionType());

    // const float means[3] = {0, 0, 0};
    // const float norms[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    // CV::ImageProcess::Config preProcessConfig;
    // ::memcpy(preProcessConfig.mean, means, sizeof(means));
    // ::memcpy(preProcessConfig.normal, norms, sizeof(norms));
    // preProcessConfig.sourceFormat = CV::BGR;
    // preProcessConfig.destFormat   = CV::BGR;
    // preProcessConfig.filterType   = CV::BILINEAR;
    // auto pretreat = std::shared_ptr<CV::ImageProcess>(CV::ImageProcess::create(preProcessConfig));
    // pretreat->convert(imgPad.data, inputSize[1], inputSize[0], 0, inputTensor);

    return 0;
}

int YolovxDet::postproc(const YolovxDet::PreprocInfo& info, std::vector<BoxInfo>& dets)
{
    // float sy = inputSize[0] / (float)image_h;
    // float sx = inputSize[1] / (float)image_w;

    float s = (float)info.imgH / inputSize[0];
    for (auto& b : dets)
    {
        float x1 = (b.x1 - info.wpad / 2) * s;
        float y1 = (b.y1 - info.hpad / 2) * s;
        float x2 = x1 + (b.x2 - b.x1) * s;
        float y2 = y1 + (b.y2 - b.y1) * s;

        b.x1 = x1;
        b.y1 = y1;
        b.x2 = x2;
        b.y2 = y2;
    }

    return 0;
}

int YolovxDet::detect(const cv::Mat& imgBgr, std::vector<BoxInfo>& dets, float score_threshold, float nms_threshold)
{
    dets.clear();

    if (imgBgr.empty()) 
    {        
        return -1;
    }

    BenchmarkTimer tm("YolovxDet");

    PreprocInfo info;
    preproc(imgBgr, info);
    tm.getDt("preproc");

    // run network
    interpreter->runSession(session);

    tm.getDt("run");

    // get output data
    MNN::Tensor *predTensors = interpreter->getSessionOutput(session, m_config.outputNode.c_str());
    // MNN::Tensor *predTensors = interpreter->getSessionOutput(session, nullptr);
    fmt::print("outputTensor: BCHW: {}x{}x{}x{}  dims={}  dimType={}\n", predTensors->batch(), predTensors->channel(), 
        predTensors->height(), predTensors->width(), predTensors->dimensions(), (int)predTensors->getDimensionType());

    MNN::Tensor predTensors_host(predTensors, predTensors->getDimensionType());
    predTensors->copyToHostTensor(&predTensors_host);

    fmt::print("outputTensor-host: BCHW: {}x{}x{}x{}  dims={}\n", predTensors_host.batch(), 
        predTensors_host.channel(), predTensors_host.height(), predTensors_host.width(), 
        predTensors_host.dimensions());

    // auto output = _Convert(*predTensors, MNN::Express::NCHW);
    // output = _Squeeze(output);

    // std::vector<int> shape = {predTensors->height(), predTensors->width()};
    // MNN::Tensor* transposedTensor = MNN::Tensor::create<float>(shape, nullptr, MNN::Tensor::TENSORFLOW);

    m_config.printSelf();

    YolovxPostproc postproc(&predTensors_host, m_config.strides, m_config.inputSize, m_config.regMax, m_config.nClass);
    std::vector<BoxInfo> boxes;
    std::vector<int> picked;
    postproc.computeBoxes(boxes, score_threshold);
    postproc.nms(boxes, picked, nms_threshold);

    for(const auto& i: picked) dets.push_back(boxes[i]);
    this->postproc(info, dets);

    tm.getDt("postproc");

    return 0;
}

static const char* class_names[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

static cv::Scalar colors[] = {
    cv::Scalar( 67,  54, 244),
    cv::Scalar( 30,  99, 233),
    cv::Scalar( 39, 176, 156),
    cv::Scalar( 58, 183, 103),
    cv::Scalar( 81, 181,  63),
    cv::Scalar(150, 243,  33),
    cv::Scalar(169, 244,   3),
    cv::Scalar(188, 212,   0),
    cv::Scalar(150, 136,   0),
    cv::Scalar(175,  80,  76),
    cv::Scalar(195,  74, 139),
    cv::Scalar(220,  57, 205),
    cv::Scalar(235,  59, 255),
    cv::Scalar(193,   7, 255),
    cv::Scalar(152,   0, 255),
    cv::Scalar( 87,  34, 255),
    cv::Scalar( 85,  72, 121),
    cv::Scalar(158, 158, 158),
    cv::Scalar(125, 139,  96)
};

int YolovxDet::drawBoxes(cv::Mat& rgb, const std::vector<BoxInfo>& boxes)
{
    for(size_t i = 0; i < boxes.size(); i++)
    {
        const BoxInfo& b = boxes[i];
        cv::Scalar color = colors[b.label % 19];

        int x1 = (int)b.x1;
        int y1 = (int)b.y1;
        int x2 = (int)b.x2;
        int y2 = (int)b.y2;

        cv::rectangle(rgb, cv::Rect(x1, y1, x2 - x1, y2 - y1), color);

        char text[256];
        sprintf(text, "%s %.2f", class_names[b.label], b.score);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = b.x1;
        int y = b.y1 - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    return 0;
}

}