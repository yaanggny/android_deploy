#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>

#include <android/log.h>

class BenchmarkTimer {
public:
    BenchmarkTimer(const char* tag, int warmup = 5, int runs = 100)
        : m_tag(tag), m_warmup(warmup), m_runs(runs) 
    {
        t0 = m_start = std::chrono::high_resolution_clock::now();
        m_times.clear();
    }
    
    void start() {
        t0 = m_start = std::chrono::high_resolution_clock::now();
    }
    
    void stop() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - m_start);
        m_times.push_back(duration.count());
    }

    double getDt(const char* tag=nullptr)
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - t0);
        double dt = duration.count() / 1000.0f;
        printf("[%s][%s]: %.4f ms\n", m_tag, tag, dt);

        t0 = end;
        return dt;
    }
    
    void reset() {
        m_times.clear();
        m_resultsPrinted = false;
    }
    
    void runBenchmark() {
        // Warm-up phase (不计入统计)
        for (int i = 0; i < m_warmup; ++i) {
            start();
            // 这里插入被测试的函数调用
            stop();
        }
        reset();
        
        // 实际测试运行
        for (int i = 0; i < m_runs; ++i) {
            start();
            // 这里插入被测试的函数调用
            stop();
        }
        
        calculateStats();
        printResults();
    }
    
    void calculateStats() {
        if (m_times.empty())
        {
            // __android_log_print(ANDROID_LOG_ERROR, "Benchmark", "[%s] No times recorded!", m_tag);
            printf("[%s] No times recorded!", m_tag);
            return;
        }
        
        // 排序以计算中位数
        std::vector<long long> sorted = m_times;
        std::sort(sorted.begin(), sorted.end());
        
        // 基础统计
        // m_min = *std::min_element(m_times.begin(), m_times.end());
        m_max = *std::max_element(m_times.begin(), m_times.end());
        
        // 计算总和
        long long sum = 0;
        for (auto t : m_times) sum += t;
        m_avg = static_cast<double>(sum) / m_times.size();
        
        // 中位数
        if (sorted.size() % 2 == 0) {
            m_median = (sorted[sorted.size()/2 - 1] + sorted[sorted.size()/2]) / 2.0;
        } else {
            m_median = sorted[sorted.size()/2];
        }
        
        // 标准差
        double variance = 0.0;
        for (auto t : m_times) {
            variance += (t - m_avg) * (t - m_avg);
        }
        variance /= m_times.size();
        m_stddev = std::sqrt(variance);
        
        m_resultsPrinted = true;
    }
    
    void printResults() const {
        if (m_times.empty()) 
        {
            // __android_log_print(ANDROID_LOG_ERROR, "Benchmark", "[%s] No times recorded!", m_tag);
            printf("[%s] No times recorded!", m_tag);
            return;
        }
        
        // __android_log_print(ANDROID_LOG_DEBUG, "Benchmark",
        printf(
            "[%25s] runs: %d  avg: %.2f ms  max: %.2f ms  "
            "med: %.2f ms  std: %.2f ms  FPS: %.1f\n",
            m_tag,
            static_cast<int>(m_times.size()),
            m_avg/1000.0,
            m_max/1000.0,
            m_median/1000.0,
            m_stddev/1000.0,
            1e6 / m_avg  // 计算帧率
        );
    }
    
    // 获取统计结果（可用于自定义输出）
    double average() const { return m_avg; }
    long long max() const { return m_max; }
    double median() const { return m_median; }
    double stddev() const { return m_stddev; }
    
    ~BenchmarkTimer() {
        if (!m_resultsPrinted && !m_times.empty()) {
            calculateStats();
            printResults();
        }
    }

private:
    const char* m_tag;
    int m_warmup;
    int m_runs;
    std::vector<long long> m_times;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start, t0;
    
    // 统计结果
    double m_avg = 0.0;
    long long m_max = 0;
    double m_median = 0.0;
    double m_stddev = 0.0;
    bool m_resultsPrinted = false;
};

// 使用示例 - 在resize_test函数中使用
// void resize_test(const cv::Mat& img, cv::Mat& resized, float downScale=2.0f, int interp=cv::INTER_LINEAR) {
//     static BenchmarkTimer timer("resize_test", 10, 100); // 预热10次，测试100次
    
//     timer.start();
    
//     cv::Size newSize(
//         static_cast<int>(img.cols / downScale),
//         static_cast<int>(img.rows / downScale)
//     );
//     cv::resize(img, resized, newSize, 0, 0, interp);
    
//     timer.stop();
    
//     // 第一次调用后自动打印结果
//     // 后续调用会累积数据，直到对象销毁时打印最终结果
// }