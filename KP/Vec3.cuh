#ifndef VEC3_CUH
#define VEC3_CUH

#include <cstdint>

enum Axis {
    X,
    Y,
    Z,
    COUNT_OF_AXIS
};

struct Vec3 {
    double data[3];

    __host__ __device__ Vec3 ();
    __host__ __device__ Vec3 (double x, double y, double z);
    __host__ __device__ Vec3 (const Vec3 &v);

    Vec3 getColor() const;

    __host__ __device__ Vec3 &operator= (const Vec3 &v);
    __host__ __device__ double &operator[] (uint64_t i);
    __host__ __device__ double operator[] (uint64_t i) const;

    __host__ __device__ Vec3 &operator+= (const Vec3 &v);
    __host__ __device__ Vec3 &operator-= (const Vec3 &v);
    __host__ __device__ Vec3 &operator*= (double el);
    __host__ __device__ Vec3 &operator*= (const Vec3 &v);
    __host__ __device__ Vec3 &operator/= (double el);
    __host__ __device__ Vec3 &operator/= (const Vec3 &v);

    friend __host__ __device__ const Vec3 operator+ (const Vec3 &v1, const Vec3 &v2);
    friend __host__ __device__ const Vec3 operator- (const Vec3 &v1, const Vec3 &v2);
    friend __host__ __device__ const Vec3 operator* (const Vec3 &v, double el);
    friend __host__ __device__ const Vec3 operator* (const Vec3 &v1, const Vec3 &v2);
    friend __host__ __device__ const Vec3 operator/ (const Vec3 &v, double el);
    friend __host__ __device__ const Vec3 operator/ (const Vec3 &v1, const Vec3 &v2);
};

__host__ __device__ void initVec3 (Vec3 &v);

__host__ __device__ Vec3 createVec3 (double x, double y, double z);

__host__ __device__ double dot (const Vec3 &a, const Vec3 &b);

__host__ __device__ double Vec3Length (const Vec3 &v);

__host__ __device__ uint32_t Vec3ToColor (const Vec3 &v);

__host__ __device__ uchar4 Vec3ToUchar4 (Vec3 a);

__host__ __device__ Vec3 uchar4ToVec3 (uchar4 a);

__host__ __device__ Vec3 colorToVec3 (uint32_t color);

__host__ __device__ Vec3 normalize (const Vec3 &v);

__host__ __device__ Vec3 prod(Vec3 a, Vec3 b);

__host__ __device__ Vec3 multiple(Vec3 a, Vec3 b, Vec3 c, Vec3 d);

__host__ __device__ Vec3 reflect(Vec3 vec, Vec3 normal);

#endif