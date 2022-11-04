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
    // std::vector<uint32_t> d = {
    //     0xA2DF4C00, 0xB4E85300, 0xA9E04C00,
    //     0xF7C9FE00, 0x99D14D00, 0xF7D1FA00,
    //     0x9ED84500, 0x92DD5600, 0xD4D0E900
    // };
    // Image tmp(3, 3, d);
    // tmp.saveToFile("test/in4.data");
    // return 0;
    Image image(input), result;
    cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
    result = std::move(image.MahalanobisDistance(data, nc));
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	fprintf(stderr, "time = %f\n", time);
    result.saveToFile(output);
    result.printInfo();
    return 0;
}