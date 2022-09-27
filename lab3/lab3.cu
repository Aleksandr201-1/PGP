#include <iostream>
#include "ImageReader.cuh"

int main () {
    std::string input, output;
    uint32_t w, h;
    std::cin >> input >> output >> w >> h;
    Image image(input), result;
    result = std::move(image.SSAA(w, h));
    result.saveToFile(output);
    return 0;
}