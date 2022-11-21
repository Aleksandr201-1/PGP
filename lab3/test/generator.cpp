#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

int main (int argc, char **argv) {
    srand(time(NULL));
    std::string  test(argv[1]), input(argv[2]), output(argv[3]);
    uint64_t limit = std::atoi(argv[4]);

    std::ofstream out(test);
    uint64_t classes = rand() % 32;
    out << input << '\n' << output << '\n' << classes << '\n';
    for (uint64_t i = 0; i < classes; ++i) {
        uint64_t pairs = rand() % 1000;
        out << pairs;
        for (uint64_t j = 0; j < pairs; ++j) {
            uint64_t tmp1 = rand() % limit;
            uint64_t tmp2 = rand() % limit;
            out << ' ' << tmp1 << ' ' << tmp2;
        }
        out << '\n';
    }
    return 0;
}