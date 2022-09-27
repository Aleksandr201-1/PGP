#ifndef IMAGE_READER_CUH
#define IMAGE_READER_CUH

#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include "GPUErrorCheck.cuh"

class Image {
    public:
        explicit Image ();
        explicit Image (const std::string &path);
        Image (Image const &image);
        Image (Image &&image);
        Image (uint32_t w, uint32_t h, const std::vector<uint32_t> &data);
        ~Image ();

        void saveToFile (const std::string &path) const;
        void readFromFile (const std::string &path);

        Image SSAA (uint32_t new_w, uint32_t new_h) const;

        friend std::ifstream &operator>> (std::ifstream &input, Image &image);
        friend std::ofstream &operator<< (std::ofstream &output, const Image &image);
        Image& operator= (const Image &image);
        Image& operator= (Image &&image);
        void printInfo () const;
    private:
        uint32_t w, h;
        std::vector<uint32_t> buff;
        //mutable texture<uint32_t, 2, cudaReadModeElementType> img_tex;
};

#endif