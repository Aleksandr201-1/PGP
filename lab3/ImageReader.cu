#include "ImageReader.cuh"

__constant__ Matrix cov_gpu[32];
__constant__ Vec3 avg_gpu[32];

void AvgCovInit (uint64_t nc, const std::vector<uint64_t> &data, const Image &img) {
    Vec3 avgRes[32];
    Matrix covRes[32];
    uint64_t idx = 0;
    for (uint64_t i = 0; i < nc; ++i) {
        initVec3(avgRes[i]);
        initMatrix(covRes[i]);
        uint64_t np = data[idx];
        for (uint64_t j = 0; j < np * 2; j += 2) {
            uint32_t color = img(data[idx + j + 2], data[idx + j + 1]);
            avgRes[i] += colorToVec3(color);
        }
        avgRes[i] /= np;
        for (uint64_t j = 0; j < np * 2; j += 2) {
            uint32_t color = img(data[idx + j + 2], data[idx + j + 1]);
            Vec3 tmp = colorToVec3(color) - avgRes[i];
            Matrix covEl;
            for (uint64_t i1 = 0; i1 < 3; ++i1) {
                for (uint64_t j1 = 0; j1 < 3; ++j1) {
                    covEl(i1, j1) = tmp[i1] * tmp[j1];
                }
            }
            covRes[i] += covEl;
        }
        if (np > 1) {
            covRes[i] /= (np - 1);
        }
        reverseMatrix(covRes[i]);
        idx += np * 2 + 1;
    }
    gpuErrorCheck(cudaMemcpyToSymbol(avg_gpu, avgRes, 32 * sizeof(Vec3)));
    gpuErrorCheck(cudaMemcpyToSymbol(cov_gpu, covRes, 32 * sizeof(Matrix)));
}

__device__ double getArg (uint32_t color, uint64_t idx) {
    double ans = 0;
    Vec3 tmp1 = colorToVec3(color) - avg_gpu[idx], tmp2;
    initVec3(tmp2);
    for (uint64_t i = 0; i < 3; ++i) {
        for (uint64_t j = 0; j < 3; ++j) {
            tmp2[i] += tmp1[j] * cov_gpu[idx](j, i);
        }
        ans += tmp1[i] * tmp2[i];
    }
    return -ans;
}

__global__ void MahalanobisKernel (uint32_t w, uint32_t h, uint64_t nc, uint32_t *data) {
    //int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //int offsetY = gridDim.y * blockDim.y;
    int offsetX = gridDim.x * blockDim.x;

    //for (uint64_t i = idy; i < h; i += offsetY) {
        for (uint64_t i = idx; i < w * h; i += offsetX) {
            uint32_t color = data[i];
            double max = getArg(color, 0);
            uint32_t curr_class = 0;
            for (uint64_t k = 1; k < nc; ++k) {
                double tmp = getArg(color, k);
                if (max <= tmp) {
                    max = tmp;
                    curr_class = k;
                }
            }
            data[i] += curr_class;
        }
    //}
}

Image::Image () {
    w = h = 0;
}

Image::Image (const std::string &path) {
    readFromFile(path);
}

Image::Image (const Image &image) : w(image.w), h(image.h), buff(image.buff) {}

Image::Image (Image &&image) : w(image.w), h(image.h), buff(std::move(image.buff)) {}

Image::Image (uint32_t w, uint32_t h, const std::vector<uint32_t> &data) : w(w), h(h) {
    if (data.size() == w * h) {
        buff = data;
    } else {
        std::cerr << "ERROR: wrong image constructor";
        exit(0);
    }
}

Image::~Image () {}

void Image::saveToFile (const std::string &path) const{
    std::ofstream output(path, std::ios::binary);
    output << *this;
}

void Image::readFromFile (const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        std::cerr << "ERROR: cant open file \"" << path.c_str() << "\"";
        exit(0);
    }
    input >> *this;
}

Image Image::MahalanobisDistance (const std::vector<uint64_t> &data, uint64_t nc) const {
    Image new_image;
    uint32_t *old_buff;
    AvgCovInit(nc, data, *this);

    gpuErrorCheck(cudaMalloc(&old_buff, sizeof(uint32_t) * w * h));
    gpuErrorCheck(cudaMemcpy(old_buff, buff.data(), sizeof(uint32_t) * w * h, cudaMemcpyHostToDevice));

    MahalanobisKernel<<<dim3(32, 32), dim3(32, 32)>>>(w, h, nc, old_buff);

    new_image.h = h;
    new_image.w = w;
    new_image.buff.resize(w * h);
    gpuErrorCheck(cudaMemcpy(&new_image.buff[0], old_buff, sizeof(uint32_t) * w * h, cudaMemcpyDeviceToHost));

    gpuErrorCheck(cudaFree(old_buff));
    return new_image;
}

const std::vector<uint32_t> &Image::getData () const {
    return buff;
}

uint32_t Image::getW () const {
    return w;
}

uint32_t Image::getH () const {
    return h;
}

std::ifstream &operator>> (std::ifstream &input, Image &image) {
    input.read(reinterpret_cast<char *>(&image.w), sizeof(image.w));
    input.read(reinterpret_cast<char *>(&image.h), sizeof(image.h));
    image.buff.resize(image.w * image.h);
    for (uint64_t i = 0; i < image.buff.size(); ++i) {
        input.read(reinterpret_cast<char *>(&image.buff[i]), sizeof(uint32_t));
    }
    return input;
}

std::ofstream &operator<< (std::ofstream &output, const Image &image) {
    output.write(reinterpret_cast<const char *>(&image.w), sizeof(image.w));
    output.write(reinterpret_cast<const char *>(&image.h), sizeof(image.h));
    for (uint64_t i = 0; i < image.buff.size(); ++i) {
        output.write(reinterpret_cast<const char *>(&image.buff[i]), sizeof(uint32_t));
    }
    return output;
}

Image& Image::operator= (const Image &image) {
    if (this == &image) {
        return *this;
    }
    w = image.w;
    h = image.h;
    buff = image.buff;
    return *this;
}

Image& Image::operator= (Image &&image) {
    if (this == &image) {
        return *this;
    }
    w = image.w;
    h = image.h;
    buff = std::move(image.buff);
    return *this;
}

uint32_t Image::operator() (uint64_t i, uint64_t j) const {
    return buff[i * w + j];
}

void Image::printInfo () const {
    std::cout << "Size: " << w << " " << h << "\n";
    std::cout << "Content:\n";
    for (uint32_t i = 0; i < h; ++i) {
        for (uint32_t j = 0; j < w; ++j) {
            std::cout << std::hex << buff[i * w + j] << " ";
        }
        std::cout << "\n";
    }
}