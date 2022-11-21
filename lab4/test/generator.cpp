#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

int main (int argc, char **argv) {
    srand(time(NULL));
    std::string output(argv[1]);
    uint64_t size = std::atoi(argv[2]);

    std::ofstream out(output);
    out << size << '\n';
    for (uint64_t i = 0; i < size; ++i) {
        for (uint64_t j = 0; j < size; ++j) {
            uint64_t tmp = rand() % 20;
            out << tmp << ' ';
        }
        out << '\n';
    }
    return 0;
}