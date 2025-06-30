//
// Created by 邓昊晴 on 14/6/2020.
//

#include "YoloV5.h"
#include "cpu.h"

bool YoloV5::hasGPU = true;
YoloV5 *YoloV5::detector = nullptr;

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

DEFINE_LAYER_CREATOR(YoloV5Focus)

YoloV5::YoloV5(const std::string& modelname_noext, bool useGPU) {
    Net = new ncnn::Net();
    // opt 需要在加载前设置
    hasGPU = ncnn::get_gpu_count() > 0;
    hasGPU = false;
    printf("use vulkan compute:  gpu-cnt=%d\n", (int)hasGPU);
    Net->opt.use_vulkan_compute = hasGPU && useGPU;  // gpu
    Net->opt.use_fp16_arithmetic = true;  // fp16运算加速
    Net->register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);

    Net->load_param((modelname_noext + ".param").c_str());
    Net->load_model((modelname_noext + ".bin").c_str());
}

YoloV5::~YoloV5() {
    delete Net;
}

std::vector<BoxInfo> YoloV5::detect(const cv::Mat& img_rgb, float threshold, float nms_threshold) 
{
    //    ncnn::Mat in_net = ncnn::Mat::from_android_bitmap_resize(env,image,ncnn::Mat::PIXEL_BGR2RGB,input_size/2,input_size/2);
    // ncnn::Mat in_net = ncnn::Mat::from_android_bitmap_resize(env, image, ncnn::Mat::PIXEL_RGBA2RGB, input_size / 2,
    //                                                          input_size / 2);
    cv::Size img_size = img_rgb.size();
    printf("img shape: %d x %d\n", img_size.height, img_size.width);
    ncnn::Mat in_net = ncnn::Mat::from_pixels_resize((unsigned char*)img_rgb.data, ncnn::Mat::PIXEL_RGB, img_rgb.cols, img_rgb.rows, input_size, input_size);
    
    float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    float mean[3] = {0, 0, 0};
    in_net.substract_mean_normalize(mean, norm);

    printf("preprocessed img size: %d x %d x %d\n", in_net.h, in_net.w, in_net.c);

    ncnn::Option opt;
    // opt.lightmode = true;
    // opt.num_threads = 4;
    opt.num_threads = ncnn::get_big_cpu_count();
    if (hasGPU)
    {
        opt.blob_allocator = &g_blob_pool_allocator;
        opt.workspace_allocator = &g_workspace_pool_allocator;
    }
    // opt.use_packing_layout = true;
    // use vulkan compute
    opt.use_vulkan_compute = hasGPU;
    
    Net->opt = opt;

    auto ex = Net->create_extractor();
    // ex.set_light_mode(true);
    // ex.set_num_threads(4);

    // ex.set_vulkan_compute(hasGPU);

    ex.input(0, in_net);
    std::vector<BoxInfo> result;
    for (const auto &layer: layers) {
        ncnn::Mat blob;
        ex.extract(layer.name.c_str(), blob);
        auto boxes = decode_infer(blob, layer.stride, {(int) img_size.width, (int) img_size.height}, input_size,
                                  num_class, layer.anchors, threshold);
        result.insert(result.begin(), boxes.begin(), boxes.end());
    }
    nms(result, nms_threshold);
    return result;
}

inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

std::vector<BoxInfo>
YoloV5::decode_infer(ncnn::Mat &data, int stride, const yolocv::YoloSize &frame_size, int net_size, int num_classes,
                     const std::vector<yolocv::YoloSize> &anchors, float threshold) {
    std::vector<BoxInfo> result;
    int grid_size = int(sqrt(data.h));
    float *mat_data[data.c];
    for (int i = 0; i < data.c; i++) {
        mat_data[i] = data.channel(i);
    }
    float cx, cy, w, h;
    for (int shift_y = 0; shift_y < grid_size; shift_y++) {
        for (int shift_x = 0; shift_x < grid_size; shift_x++) {
            int loc = shift_x + shift_y * grid_size;
            for (int i = 0; i < 3; i++) {
                float *record = mat_data[i];
                float *cls_ptr = record + 5;
                for (int cls = 0; cls < num_classes; cls++) {
                    float score = sigmoid(cls_ptr[cls]) * sigmoid(record[4]);
                    if (score > threshold) {
                        cx = (sigmoid(record[0]) * 2.f - 0.5f + (float) shift_x) * (float) stride;
                        cy = (sigmoid(record[1]) * 2.f - 0.5f + (float) shift_y) * (float) stride;
                        w = pow(sigmoid(record[2]) * 2.f, 2) * anchors[i].width;
                        h = pow(sigmoid(record[3]) * 2.f, 2) * anchors[i].height;
                        //printf("[grid size=%d, stride = %d]x y w h %f %f %f %f\n",grid_size,stride,record[0],record[1],record[2],record[3]);
                        BoxInfo box;
                        box.x1 = std::max(0, std::min(frame_size.width, int((cx - w / 2.f) * (float) frame_size.width / (float) net_size)));
                        box.y1 = std::max(0, std::min(frame_size.height, int((cy - h / 2.f) * (float) frame_size.height / (float) net_size)));
                        box.x2 = std::max(0, std::min(frame_size.width, int((cx + w / 2.f) * (float) frame_size.width / (float) net_size)));
                        box.y2 = std::max(0, std::min(frame_size.height, int((cy + h / 2.f) * (float) frame_size.height / (float) net_size)));
                        box.score = score;
                        box.label = cls;
                        result.push_back(box);
                    }
                }
            }
            for (auto &ptr:mat_data) {
                ptr += (num_classes + 5);
            }
        }
    }
    return result;
}

void YoloV5::nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH) {
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}
