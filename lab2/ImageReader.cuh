#ifndef IMAGE_READER_CUH
#define IMAGE_READER_CUH

// #ifdef __CUDACC__
// #define CUDA_CALLABLE_MEMBER __host__ __device__
// //#define CUDA_CALLABLE_MEMBER __global__
// #else
// #define CUDA_CALLABLE_MEMBER
// #endif 

#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <iostream>

class Image {
    //using uint32_t = uint32_t;
    //public:
        //static CUDA_CALLABLE_MEMBER void SSAA (uint32_t old_w, uint32_t old_h, uint32_t *old_buff, uint32_t new_w, uint32_t new_h, uint32_t *new_buff);
    public:
        Image ();
        Image (const std::string &path);
        Image (uint32_t w, uint32_t h, const std::vector<uint32_t> &data);
        ~Image ();
        void saveToFile (const std::string &path) const;
        void readFromFile (const std::string &path);
        Image SSAA (uint32_t new_w, uint32_t new_h) const;

        friend std::ifstream &operator>> (std::ifstream &input, Image &image);
        friend std::ofstream &operator<< (std::ofstream &output, const Image &image);
        Image& operator= (const Image& image);
        void printInfo () const;
    private:
        uint32_t w, h;
        std::vector<uint32_t> buff;
};

#endif