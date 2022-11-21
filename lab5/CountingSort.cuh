#ifndef COUNTING_SORT_CUH
#define COUNTING_SORT_CUH

#include <cstdint>
#include <vector>
#include "GPUErrorCheck.cuh"

using DATA_TYPE = uint32_t;
const uint64_t DATA_MAX_EL = (2ll << 24) - 1;// (DATA_TYPE)(-1);

class Array {
    public:
        explicit Array ();
        Array (Array const &array);
        Array (Array &&array);
        Array (uint32_t n, const std::vector<DATA_TYPE> &data);
        ~Array ();

        void CountingSort ();

        friend std::istream &operator>> (std::istream &input, Array &image);
        friend std::ostream &operator<< (std::ostream &output, const Array &image);
        Array& operator= (const Array &image);
        Array& operator= (Array &&image);
    private:
        uint32_t n;
        std::vector<DATA_TYPE> data;
};

#endif