#include "yolov5.h"

#include <gpu.h>

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

DEFINE_LAYER_CREATOR(YoloV5Focus)

YoloV5::YoloV5(const std::string& model_path_noext, int _model_size, bool use_gpu)
{
    net.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    ncnn::Option opt;
    // opt.lightmode = true;
    // opt.use_packing_layout = true;
    opt.num_threads = ncnn::get_big_cpu_count();
    if(use_gpu)
    {
#if NCNN_VULKAN
        bool hasGPU = ncnn::get_gpu_count() > 0;  // get_gpu_count is available vulkan version ncnn only
        opt.use_vulkan_compute = use_gpu && hasGPU;
#endif
        opt.blob_allocator = &g_blob_pool_allocator;
        opt.workspace_allocator = &g_workspace_pool_allocator;
    }
    net.opt = opt;

    net.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);

    model_size = _model_size;
    inH = inW = model_size;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", model_path_noext.c_str());
    sprintf(modelpath, "%s.bin", model_path_noext.c_str());

    net.load_param(parampath);
    net.load_model(modelpath);

    meanVals[0] = 0.f;
    meanVals[1] = 0.f;
    meanVals[2] = 0.f;

    stdVals[0] = 1.f / 255.f;
    stdVals[1] = 1.f / 255.f;
    stdVals[2] = 1.f / 255.f;    
}

void printSize(const cv::Mat& img, const char* ext)
{
    printf("[%s]:  imgH: %d, imgW: %d\n", ext, img.rows, img.cols);
}

void YoloV5::preproc(const cv::Mat &img, cv::Mat &out, int& ph, int& pw)
{
    // resize
    int ho = model_size;
    int wo = model_size;
    if(imgH > imgW)
    {
        wo = model_size * imgW / imgH;
    }
    else
    {
        ho = model_size * imgH / imgW;
    }

    cv::Mat imgRz;
    cv::resize(img, imgRz, cv::Size(wo, ho));
    // printSize(imgRz, "imgRz");

    // pad
    // int 
    ph = model_size - ho;
    // int 
    pw = model_size - wo;
    cv::copyMakeBorder(imgRz, out, ph/2, ph - ph/2, pw/2, pw - pw/2, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    // printSize(out, "out");
}
