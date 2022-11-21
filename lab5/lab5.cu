#include <iostream>
#include "CountingSort.cuh"


int main () {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Array in;
    std::cin >> in;
    in.CountingSort();
    std::cout << std::hex << in;

    return 0;
}