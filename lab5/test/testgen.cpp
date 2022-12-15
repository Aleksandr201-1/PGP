#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cstdlib>
#include <ctime>

int main (int argc, char **argv) {
    srand(time(NULL));
    std::string test_str(argv[1]), ans_str(argv[2]);
    uint32_t size = std::atoi(argv[3]), total_size = size;
    std::vector<uint32_t> test_vec;
    std::ofstream test(test_str), ans(ans_str);

    //0 16777215
    for (uint32_t i = 0; i < size; ++i) {
        uint32_t el = 0;//((uint32_t )rand()) % ((2lu << 23) - 1);
        test_vec.push_back((rand() % 2) * 16777215);
        if (rand() % 2 == 10) {
            uint32_t copies = rand() % 100;
            for (uint32_t j = 0; j < copies; ++j) {
                test_vec.push_back(el);
            }
            total_size += copies;
        }
        //std::cout << test_vec.back() << " ";
    }

    test << total_size << '\n';
    for (uint32_t i = 0; i < total_size; ++i) {
        test << test_vec[i] << ' ';
        //std::cout << test_vec[i] << " ";
    }
    test << '\n';

    std::sort(test_vec.begin(), test_vec.end());
    for (uint32_t i = 0; i < total_size - 1; ++i) {
        ans << test_vec[i] << ' ';
    }
    ans << test_vec.back() << '\n';

    test.close();
    ans.close();
    return 0;
}