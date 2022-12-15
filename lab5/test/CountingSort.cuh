#ifndef COUNTING_SORT_CUH
#define COUNTING_SORT_CUH

#include <cstdint>
#include <vector>
#include "GPUErrorCheck.cuh"

using DATA_TYPE = uint32_t;
const uint32_t DATA_SIZE = sizeof(DATA_TYPE) * 8;
const uint32_t DATA_MAX_EL = (2lu << 23) - 1;// (DATA_TYPE)(-1);

class Array {
    private:
        //void histogram ();
        void scan (uint32_t *count, uint32_t size);
    public:
        explicit Array ();
        Array (Array const &array);
        Array (Array &&array);
        Array (const std::vector<DATA_TYPE> &data);
        ~Array ();

        uint64_t CountingSortGPU ();
        uint64_t CountingSortCPU ();

        friend std::istream &operator>> (std::istream &input, Array &image);
        friend std::ostream &operator<< (std::ostream &output, const Array &image);
        Array& operator= (const Array &image);
        Array& operator= (Array &&image);
    private:
        std::vector<DATA_TYPE> data;
};

#endif