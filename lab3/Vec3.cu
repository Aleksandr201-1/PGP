#include "Vec3.cuh"

__host__ __device__ Vec3 &Vec3::operator= (const Vec3 &v) {
    for (uint64_t i = 0; i < 3; ++i) {
        data[i] = v.data[i];
    }
    return *this;
}

__host__ __device__ double &Vec3::operator[] (uint64_t i) {
    return data[i];
}

__host__ __device__ double Vec3::operator[] (uint64_t i) const {
    return data[i];
}

__host__ __device__ Vec3 &Vec3::operator+= (const Vec3 &v) {
    for (uint64_t i = 0; i < 3; ++i) {
        data[i] += v.data[i];
    }
    return *this;
}

__host__ __device__ Vec3 &Vec3::operator-= (const Vec3 &v) {
    for (uint64_t i = 0; i < 3; ++i) {
        data[i] -= v.data[i];
    }
    return *this;
}

__host__ __device__ Vec3 &Vec3::operator*= (double el) {
    for (uint64_t i = 0; i < 3; ++i) {
        data[i] *= el;
    }
    return *this;
}

__host__ __device__ Vec3 &Vec3::operator/= (double el) {
    for (uint64_t i = 0; i < 3; ++i) {
        data[i] /= el;
    }
    return *this;
}

__host__ __device__ const Vec3 operator+ (const Vec3 &v1, const Vec3 &v2) {
    Vec3 ans;
    for (uint64_t i = 0; i < 3; ++i) {
        ans.data[i] = v1.data[i] + v2.data[i];
    }
    return ans;
}

__host__ __device__ const Vec3 operator- (const Vec3 &v1, const Vec3 &v2) {
    Vec3 ans;
    for (uint64_t i = 0; i < 3; ++i) {
        ans.data[i] = v1.data[i] - v2.data[i];
    }
    return ans;
}

__host__ __device__ const Vec3 operator* (const Vec3 &v, double el) {
    Vec3 ans;
    for (uint64_t i = 0; i < 3; ++i) {
        ans.data[i] = v.data[i] * el;
    }
    return ans;
}

__host__ __device__ const Vec3 operator/ (const Vec3 &v, double el) {
    Vec3 ans;
    for (uint64_t i = 0; i < 3; ++i) {
        ans.data[i] = v.data[i] / el;
    }
    return ans;
}

__host__ __device__ void initVec3 (Vec3 &c) {
    for (int i = 0; i < 3; ++i) {
        c.data[i] = 0;
    }
}

__host__ __device__ Vec3 colorToVec3 (uint32_t color) {
    Vec3 ans;
    for (uint64_t i = 0; i < 3; ++i) {
        ans[i] = (color >> (24 - 8 * i)) & 255;
    }
    return ans;
}