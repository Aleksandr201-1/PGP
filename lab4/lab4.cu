#include <iostream>
#include "Matrix.cuh"

int main () {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.precision(10);
    std::cout.setf(std::ios::scientific);

    uint64_t n;
    std::cin >> n;
    Matrix input(n), output;
    std::cin >> input;
    output = std::move(input.reverse());
    std::cout << output;
    return 0;
}