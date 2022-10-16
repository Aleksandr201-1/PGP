#include <iostream>
#include <chrono>
#include "ImageReader.cuh"

int main () {
    std::string input, output;
    uint32_t w, h;
    std::cin >> input >> output >> w >> h;
    Image image(input), result;

    // auto result1 = std::move(image.SSAAcpu(w, h));
    // uint64_t cpuTime = result1.second;
    // result1.first.saveToFile(output + "cpu.data");

    auto result2 = std::move(image.SSAAgpu(w, h));
    float gpuTime = result2.second;
    result2.first.saveToFile(output + "gpu.data");

    //std::cout << "CPU time: " << cpuTime << "mcs\n";
    std::cout << "GPU time: " << gpuTime * 1000 << "mcs\n";

    return 0;
}