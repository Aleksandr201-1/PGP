#include <iostream>
#include <chrono>
#include "Matrix.cuh"

using duration_t = std::chrono::microseconds;

int main () {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.precision(10);
    std::cout.setf(std::ios::scientific);

    uint64_t n;
    std::cin >> n;
    Matrix input(n);
    std::cin >> input;
    auto outputCPU = std::move(input.reverseCPU());
    auto outputGPU = std::move(input.reverseGPU());
    std::cout << "\ncpu time: " << outputCPU.first << "\n";
    std::cout << "\ngpu time: " << outputGPU.first << "\n";
    //std::cout << output;
    //std::cout << '\n' << input * output;
    return 0;
}