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
    Image image(input), result;
    result = std::move(image.MahalanobisDistance(data, nc));
    result.saveToFile(output);
    
    std::cout << "Out\n";
    result.printInfo();
    std::cout << "\n";
    return 0;
}