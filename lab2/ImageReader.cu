#include "ImageReader.cuh"

texture<uint32_t, 2, cudaReadModeElementType> img_tex;

//__global__ void SSAAkernel (uint32_t old_w, uint32_t old_h, texture<uint32_t, 2, cudaReadModeElementType> img_tex, uint32_t new_w, uint32_t new_h, uint32_t *new_buff) {
//     uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     uint32_t offset = blockDim.x * gridDim.x;

//     uint32_t block_w = old_w / new_w, block_h = old_h / new_h;
//     uint32_t block_size = block_w * block_h, block_count = new_h * new_w;
//     while (idx < block_count) {
//         uint32_t i0 = idx / new_w, j0 = idx % new_w;
//         uint32_t r = 0, g = 0, b = 0, a = 0;
//         for (uint32_t i1 = 0; i1 < block_h; ++i1) {
//             for (uint32_t j1 = 0; j1 < block_w; ++j1) {
//                 uint32_t color = tex2D(old_buff, i1 * old_w + i0 * block_size * new_w, j1 + j0 * block_w);
//                 r += (color >> 24) & 255;
//                 g += (color >> 16) & 255;
//                 b += (color >> 8)  & 255;
//                 a += color  & 255;
//             }
//         }
//         r = (r / block_size) << 24;
//         g = (g / block_size) << 16;
//         b = (b / block_size) << 8;
//         a = a / block_size;
//         new_buff[i0 * new_w + j0] = r + g + b + a;
//         idx += offset;
//     }
// }

__global__ void SSAAkernel (uint32_t old_w, uint32_t old_h, uint32_t new_w, uint32_t new_h, uint32_t *new_buff) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t offset = blockDim.x * gridDim.x;

    uint32_t block_w = old_w / new_w, block_h = old_h / new_h;
    uint32_t block_size = block_w * block_h, block_count = new_h * new_w;
    while (idx < block_count) {
        uint32_t i0 = idx / new_w, j0 = idx % new_w;
        uint32_t r = 0, g = 0, b = 0, a = 0;
        for (uint32_t i1 = 0; i1 < block_h; ++i1) {
            for (uint32_t j1 = 0; j1 < block_w; ++j1) {
                uint32_t color = tex2D(img_tex, j1 + j0 * block_w, i1 + i0 * block_h);
                //uint32_t color = tex1D(img_tex, i1 * old_w + i0 * block_size * new_w + j1 + j0 * block_w);
                //printf("%ld ", color);
                r += (color >> 24) & 255;
                g += (color >> 16) & 255;
                b += (color >> 8)  & 255;
                a += color  & 255;
            }
        }
        r = (r / block_size) << 24;
        g = (g / block_size) << 16;
        b = (b / block_size) << 8;
        a = a / block_size;
        new_buff[i0 * new_w + j0] = r + g + b + a;
        idx += offset;
    }
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
        fprintf(stderr, "ERROR: wrong image constructor");
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
        fprintf(stderr, "ERROR: cant open file \"%s\"", path.c_str());
        exit(0);
    }
    input >> *this;
}

Image Image::SSAA (uint32_t new_w, uint32_t new_h) const {
    Image new_image;
    uint32_t *new_buff;
    cudaArray *old_buff;
    //texture<uint32_t, 2, cudaReadModeElementType> img_tex;
    cudaChannelFormatDesc channel = cudaCreateChannelDesc<uint32_t>();

    gpuErrorCheck(cudaMallocArray(&old_buff, &channel, w, h));
    //gpuErrorCheck(cudaMemcpyToArray(old_buff, 0, 0, buff.data(), sizeof(uint32_t) * w * h, cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy2DToArray(old_buff, 0, 0, buff.data(), sizeof(uint32_t) * w, sizeof(uint32_t) * w, h, cudaMemcpyHostToDevice));

    img_tex.addressMode[0] = cudaAddressModeClamp;
    img_tex.addressMode[1] = cudaAddressModeClamp;
    img_tex.channelDesc = channel;
    img_tex.filterMode = cudaFilterModePoint;
    img_tex.normalized = false;

    gpuErrorCheck(cudaBindTextureToArray(img_tex, old_buff, channel));
    //gpuErrorCheck(cudaMalloc(&old_buff, sizeof(uint32_t) * w * h));
    gpuErrorCheck(cudaMalloc(&new_buff, sizeof(uint32_t) * new_w * new_h));
    //gpuErrorCheck(cudaMemcpy(old_buff, buff.data(), sizeof(uint32_t) * w * h, cudaMemcpyHostToDevice));

    //SSAAkernel<<<1024, 1024>>>(w, h, img_tex, new_w, new_h, new_buff);
    SSAAkernel<<<1024, 1024>>>(w, h, new_w, new_h, new_buff);
    gpuErrorCheck(cudaGetLastError());
    gpuErrorCheck(cudaDeviceSynchronize());

    new_image.h = new_h;
    new_image.w = new_w;
    new_image.buff.resize(new_w * new_h);
    gpuErrorCheck(cudaMemcpy(&new_image.buff[0], new_buff, sizeof(uint32_t) * new_w * new_h, cudaMemcpyDeviceToHost));

    gpuErrorCheck(cudaUnbindTexture(img_tex));
    gpuErrorCheck(cudaFreeArray(old_buff));
    gpuErrorCheck(cudaFree(new_buff));
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