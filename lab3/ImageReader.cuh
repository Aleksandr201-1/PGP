#ifndef IMAGE_READER_CUH
#define IMAGE_READER_CUH

#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include "GPUErrorCheck.cuh"
#include "Matrix.cuh"
#include "Vec3.cuh"

class Image {
    private:
        void swapData ();
    public:
        explicit Image ();
        explicit Image (const std::string &path);
        Image (Image const &image);
        Image (Image &&image);
        Image (uint32_t w, uint32_t h, const std::vector<uint32_t> &data);
        ~Image ();

        void saveToFile (const std::string &path) const;
        void readFromFile (const std::string &path);

        Image MahalanobisDistance (const std::vector<uint64_t> &data, uint64_t nc) const;
        const std::vector<uint32_t> &getData () const;
        uint32_t getW () const;
        uint32_t getH () const;

        friend std::ifstream &operator>> (std::ifstream &input, Image &image);
        friend std::ofstream &operator<< (std::ofstream &output, const Image &image);
        Image& operator= (const Image &image);
        Image& operator= (Image &&image);
        uint32_t operator() (uint64_t i, uint64_t j) const;
        void printInfo () const;
    private:
        uint32_t w, h;
        std::vector<uint32_t> buff;
};

#endif