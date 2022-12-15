#include <iostream>
#include "CountingSort.cuh"


int main () {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Array in;
    std::cin >> in;
    auto time = in.CountingSortGPU();
    //std::cout << std::hex << in;
    //std::cout << in;

    std::cout << "time: " << time << "\n";

    return 0;
}