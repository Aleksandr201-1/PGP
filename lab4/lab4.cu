#include <iostream>
#include "Matrix.cuh"

int main () {
    uint64_t n;
    std::cin >> n;
    Matrix input(n), output;
    std::cin >> input;
    output = std::move(input.reverse());
    output.printMatrix();
    std::cout << input * output;
    return 0;
}