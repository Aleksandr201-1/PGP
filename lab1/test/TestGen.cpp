#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>

void generateTest () {}

void generateAns () {}

void help () {
    std::cout << "Usage: ./TestGen [--key] {arg}\n"
    << "-h, --help \t\t\tPrint usage\n"
    << "-t, --test [filename]\t\t\tWrite test to [filename]\n"
    << "-a, --ans [filename]\t\t\tWrite answer to [filename]\n"
    << "-s, --size [int1] [int2]\t\t\t[int1] - count of vectors, [int2] - size of vectors\n";
}

void error (char *name, uint64_t code) {
    std::cerr << "Incorrect input format. Use \"" << name << "--help\"\n";
    exit(code);
}

int main (int argc, char **argv) {
    std::srand(std::time(NULL));
    uint64_t size = 0, count = 0;
    std::string in, out;
    for (uint64_t i = 0; i < argc; ++i) {
        std::string str(argv[i]);
        if (str == "-h" || str == "--help") {
            help();
            return 0;
        } else if (str == "-t" || str == "--test") {
            if (i < argc - 1) {
                in = std::string(argv[i + 1]);
                ++i;
            } else {
                error(argv[0], 1);
            }
        } else if (str == "-a" || str == "--ans") {
            if (i < argc - 1) {
                out = std::string(argv[i + 1]);
                ++i;
            } else {
                error(argv[0], 2);
            }
        } else if (str == "-s" || str == "--size") {
            if (i < argc - 2) {
                count = std::stoll(argv[i + 1]);
                size = std::stoll(argv[i + 2]);
                i += 2;
            } else {
                error(argv[0], 3);
            }
        }
    }

    if (in.empty() || out.empty() || size == 0 || count == 0) {
        error(argv[0], 4);
    }

    std::ofstream test(in), ans(out);
    std::vector<uint64_t> vec2;
    test << size << "\n";
    for (uint64_t i = 0; i < size; ++i) {
        uint64_t n1, n2, a;
        n1 = std::rand() % 1000;
        n2 = std::rand() % 1000;
        a = std::max(n1, n2);
        test << n1 << " ";
        vec2.push_back(n2);
        ans << a << " ";
    }
    test << "\n";
    for (auto el : vec2) {
        test << el << " ";
    }

    test.close();
    ans.close();
    return 0;
}