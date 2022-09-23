#include <cstdlib>
#include <string>
#include "ImageReader.cuh"

int main () {
    std::string input, output;
    uint32_t w, h;
    std::cin >> input >> output >> w >> h;
    Image image(input), result;
    result = image.SSAA(w, h);
    result.saveToFile(output);

    image.printInfo();
    std::cout << "\n";
    result.printInfo();

    // std::vector<uint32_t> vec = {0x01020300, 0x04050600, 0x07080900, 0x0A0B0C00,
    //                              0x0D0E0F00, 0x10111200, 0x13141500, 0x16171800,
    //                              0x191A1B00, 0x1C1D1E00, 0x1F202100, 0x22232400,
    //                              0x25262700, 0x28292A00, 0x2B2C2D00, 0x2E2F3000};
    //Image i(4,4, vec);
    //i.saveToFile("in2.data");

    //Image img("in2.data"), img2;
    //img.print();
    //img.SSAA(2, 4).print();
    //img.saveToFile("in.data");
    //std::cout << "Done\n";
    //img2.print();

    return 0;
}