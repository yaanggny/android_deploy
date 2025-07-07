#ifndef MODEL_BASE_H
#define MODEL_BASE_H

#include <fmt/printf.h>
#include <fmt/ranges.h>
#include <fmt/format.h>

#include <string>
#include <vector>

struct ModelConfig
{
    int inputSize;
    int regMax;
    int nClass;
    bool hasPermute;
    std::vector<int> strides;
    std::string modelname_noext;
    std::string inputNode;
    std::string outputNode;

    ModelConfig() : inputSize(320), regMax(7), nClass(80), hasPermute(true) {}
    
    void printSelf() const
    {
        fmt::print("ModelConfig: {}  inputSz={}  hasPermute={}  nc={}  regMax={}\n\tstrides={}  inputNode=[{}]  outputNode=[{}]\n",
            modelname_noext, inputSize, (int)hasPermute, nClass, regMax, strides, inputNode, outputNode);
    }
};

#endif /* MODEL_BASE_H */
