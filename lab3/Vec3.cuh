#ifndef VEC3_CUH
#define VEC3_CUH

#include <cstdint>

struct Vec3 {
    double data[3];
    __host__ __device__ Vec3 &operator= (const Vec3 &v);
    __host__ __device__ double &operator[] (uint64_t i);
    __host__ __device__ double operator[] (uint64_t i) const;

    __host__ __device__ Vec3 &operator+= (const Vec3 &v);
    __host__ __device__ Vec3 &operator-= (const Vec3 &v);
    __host__ __device__ Vec3 &operator*= (double el);
    __host__ __device__ Vec3 &operator/= (double el);

    friend __host__ __device__ const Vec3 operator+ (const Vec3 &v1, const Vec3 &v2);
    friend __host__ __device__ const Vec3 operator- (const Vec3 &v1, const Vec3 &v2);
    friend __host__ __device__ const Vec3 operator* (const Vec3 &v, double el);
    friend __host__ __device__ const Vec3 operator/ (const Vec3 &v, double el);
};

__host__ __device__ void initVec3 (Vec3 &c);

__host__ __device__ Vec3 colorToVec3 (uint32_t color);

#endif