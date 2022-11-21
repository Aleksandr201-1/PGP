#include <iostream>
#include "ImageReader.cuh"

int main () {
    std::string input, output;
    uint32_t nc;
    std::cin >> input >> output >> nc;
    std::vector<uint64_t> data;
    for (uint64_t i = 0; i < nc; ++i) {
        uint64_t np;
        std::cin >> np;
        data.push_back(np);
        for (uint64_t j = 0; j < np; ++j) {
            uint64_t a, b;
            std::cin >> a >> b;
            data.push_back(a);
            data.push_back(b);
        }
    }
    std::cout << "q\n";
    Image image(input);
    std::cout << "q\n";
    auto resultGPU = image.MahalanobisDistanceCPU(data, nc);
    std::cout << "GPU time: " << resultGPU.second << '\n';

    //auto resultCPU = image.MahalanobisDistanceCPU(data, nc);
    //std::cout << "CPU time: " << resultCPU.second << '\n';

    return 0;
}