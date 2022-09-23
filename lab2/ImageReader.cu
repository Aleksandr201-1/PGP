#include "ImageReader.cuh"

__global__ void SSAAkernel (uint32_t old_w, uint32_t old_h, const uint32_t *old_buff, uint32_t new_w, uint32_t new_h, uint32_t *new_buff) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t offset = blockDim.x * gridDim.x;

    uint32_t w = old_w / new_w, h = old_h / new_h;
    uint32_t block_size = w * h, block_count = new_h * new_w;
    while (idx < block_count) {
        uint32_t i0 = idx / new_w, j0 = idx % new_w;
        uint32_t r = 0, g = 0, b = 0, a = 0;
        for (uint32_t i1 = 0; i1 < h; ++i1) {
            for (uint32_t j1 = 0; j1 < w; ++j1) {
                uint32_t color = old_buff[i1 * old_w + i0 * block_size * w + j1 + j0 * w];
                r += (color >> 24) & 255;
                g += (color >> 16) & 255;
                b += (color >> 8)  & 255;
                a += (color + 0)  & 255;
            }
        }
        r = (r / block_size) << 24;
        g = (g / block_size) << 16;
        b = (b / block_size) << 8;
        a = (a / block_size);
        new_buff[i0 * new_w + j0] = r + g + b + a;
        printf("0x%08x ", new_buff[i0 * new_w + j0]);
        idx += offset;
    }
}

void SS (uint32_t old_w, uint32_t old_h, const uint32_t *old_buff, uint32_t new_w, uint32_t new_h, uint32_t *new_buff) {
    //Image::SSAA(old_w, old_h, old_buff, new_w, new_h, new_buff);
    //printf("q");
    //uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    //uint32_t offset = blockDim.x * gridDim.x;

    uint32_t w = old_w / new_w, h = old_h / new_h;
    uint32_t block_size = w * h;
    for (uint32_t i0 = 0; i0 < new_h; ++i0) {
        for (uint32_t j0 = 0; j0 < new_w; ++j0) {
            uint32_t r = 0, g = 0, b = 0, a = 0;
            for (uint32_t i1 = 0; i1 < h; ++i1) {
                for (uint32_t j1 = 0; j1 < w; ++j1) {
                    uint32_t color = old_buff[i1 * old_w + i0 * block_size * w + j1 + j0 * w];
                    printf("0x%08x ", color);
                    r += (color >> 24) & 255;
                    g += (color >> 16) & 255;
                    b += (color >> 8)  & 255;
                    a += (color + 0)  & 255;
                }
                printf("\n");
            }
            r = (r / block_size) << 24;
            g = (g / block_size) << 16;
            b = (b / block_size) << 8;
            a = (a / block_size);
            new_buff[i0 * new_w + j0] = r + g + b + a;
        }
    }
}

Image::Image () {
    w = h = 0;
}

Image::Image (const std::string &path) {
    readFromFile(path);
}

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
    input >> *this;
}

Image Image::SSAA (uint32_t new_w, uint32_t new_h) const {
    Image new_image;
    uint32_t *old_buff, *new_buff;
    uint32_t *vv = (uint32_t *)malloc(sizeof(uint32_t) * new_w * new_h);

    cudaMalloc(&old_buff, sizeof(uint32_t) * w * h);
    cudaMalloc(&new_buff, sizeof(uint32_t) * new_w * new_h);

    cudaMemcpy(old_buff, buff.data(), sizeof(uint32_t) * w * h, cudaMemcpyHostToDevice);

    SSAAkernel<<<1024, 1024>>>(w, h, old_buff, new_w, new_h, new_buff);
    //SSAAkernel(w, h, buff.data(), new_w, new_h, vv);

    new_image.h = new_h;
    new_image.w = new_w;
    new_image.buff.resize(new_w * new_h);
    cudaMemcpy(&new_image.buff[0], new_buff, sizeof(uint32_t) * new_w * new_h, cudaMemcpyDeviceToHost);
    // for (uint32_t i = 0; i < new_h; ++i) {
    //     for (uint32_t j = 0; j < new_w; ++j) {
    //         new_image.buff[i * new_w + j] = vv[i * new_w + j];
    //     }
    // }

    cudaFree(old_buff);
    cudaFree(new_buff);
    free(vv);
    return new_image;
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

Image& Image::operator= (const Image& image) {
    if (this == &image) {
        return *this;
    }
    w = image.w;
    h = image.h;
    buff = image.buff;
    return *this;
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