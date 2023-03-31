#include "Vec3.cuh"

__host__ __device__ Vec3::Vec3 () {
    for (uint64_t i = 0; i < COUNT_OF_AXIS; ++i) {
        data[i] = 0;
    }
}

__host__ __device__ Vec3::Vec3 (double x, double y, double z) {
    data[X] = x;
    data[Y] = y;
    data[Z] = z;
}

__host__ __device__ Vec3::Vec3 (const Vec3 &v) {
    for (uint64_t i = 0; i < COUNT_OF_AXIS; ++i) {
        data[i] = v[i];
    }
}

Vec3 Vec3::getColor () const {
    return createVec3(data[X] * 255, data[Y] * 255, data[Z] * 255);
}

__host__ __device__ Vec3 &Vec3::operator= (const Vec3 &v) {
    for (uint64_t i = 0; i < COUNT_OF_AXIS; ++i) {
        data[i] = v[i];
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
    for (uint64_t i = 0; i < COUNT_OF_AXIS; ++i) {
        data[i] += v[i];
    }
    return *this;
}

__host__ __device__ Vec3 &Vec3::operator-= (const Vec3 &v) {
    for (uint64_t i = 0; i < COUNT_OF_AXIS; ++i) {
        data[i] -= v[i];
    }
    return *this;
}

__host__ __device__ Vec3 &Vec3::operator*= (double el) {
    for (uint64_t i = 0; i < COUNT_OF_AXIS; ++i) {
        data[i] *= el;
    }
    return *this;
}

__host__ __device__ Vec3 &Vec3::operator*= (const Vec3 &v) {
    for (uint64_t i = 0; i < COUNT_OF_AXIS; ++i) {
        data[i] *= v[i];
    }
    return *this;
}

__host__ __device__ Vec3 &Vec3::operator/= (double el) {
    for (uint64_t i = 0; i < COUNT_OF_AXIS; ++i) {
        data[i] /= el;
    }
    return *this;
}

__host__ __device__ Vec3 &Vec3::operator/= (const Vec3 &v) {
    for (uint64_t i = 0; i < COUNT_OF_AXIS; ++i) {
        data[i] /= v[i];
    }
    return *this;
}

__host__ __device__ const Vec3 operator+ (const Vec3 &v1, const Vec3 &v2) {
    Vec3 ans;
    for (uint64_t i = 0; i < COUNT_OF_AXIS; ++i) {
        ans[i] = v1[i] + v2[i];
    }
    return ans;
}

__host__ __device__ const Vec3 operator- (const Vec3 &v1, const Vec3 &v2) {
    Vec3 ans;
    for (uint64_t i = 0; i < COUNT_OF_AXIS; ++i) {
        ans[i] = v1[i] - v2[i];
    }
    return ans;
}

__host__ __device__ const Vec3 operator* (const Vec3 &v, double el) {
    Vec3 ans;
    for (uint64_t i = 0; i < COUNT_OF_AXIS; ++i) {
        ans[i] = v[i] * el;
    }
    return ans;
}

__host__ __device__ const Vec3 operator* (const Vec3 &v1, const Vec3 &v2) {
    Vec3 ans;
    for (uint64_t i = 0; i < COUNT_OF_AXIS; ++i) {
        ans[i] = v1[i] * v2[i];
    }
    return ans;
}

__host__ __device__ const Vec3 operator/ (const Vec3 &v, double el) {
    Vec3 ans;
    for (uint64_t i = 0; i < COUNT_OF_AXIS; ++i) {
        ans[i] = v[i] / el;
    }
    return ans;
}

__host__ __device__ const Vec3 operator/ (const Vec3 &v1, const Vec3 &v2) {
    Vec3 ans;
    for (uint64_t i = 0; i < COUNT_OF_AXIS; ++i) {
        ans[i] = v1[i] / v2[i];
    }
    return ans;
}

__host__ __device__ void initVec3 (Vec3 &v) {
    for (int i = 0; i < COUNT_OF_AXIS; ++i) {
        v[i] = 0;
    }
}

__host__ __device__ Vec3 createVec3 (double x, double y, double z) {
    Vec3 v;
    v[X] = x;
    v[Y] = y;
    v[Z] = z;
    return v;
}

__host__ __device__ double dot (const Vec3 &a, const Vec3 &b) {
    double ans = 0;
    for (uint64_t i = 0; i < COUNT_OF_AXIS; ++i) {
        ans += a[i] * b[i];
    }
    return ans;
}

__host__ __device__ double Vec3Length (const Vec3 &v) {
    return std::sqrt(dot(v, v));
}

__host__ __device__ uint32_t Vec3ToColor (const Vec3 &v) {
    uint32_t color = 0;
    for (uint64_t i = 0; i < COUNT_OF_AXIS; ++i) {
        color += ((uint32_t)v[i] >> (24 - 8 * i)) & 255;
    }
    return color;
}

__host__ __device__ uchar4 Vec3ToUchar4 (Vec3 a) {
    return make_uchar4(a[X], a[Y], a[Z], 0);
}

__host__ __device__ Vec3 uchar4ToVec3 (uchar4 a) {
    return createVec3(a.x * 1.0, a.y * 1.0, a.z * 1.0);
}

__host__ __device__ Vec3 colorToVec3 (uint32_t color) {
    Vec3 ans;
    for (uint64_t i = 0; i < COUNT_OF_AXIS; ++i) {
        ans[i] = (color >> (24 - 8 * i)) & 255;
    }
    return ans;
}

__host__ __device__ Vec3 normalize (const Vec3 &v) {
    Vec3 ans;
    double length = Vec3Length(v);
    for (uint64_t i = 0; i < COUNT_OF_AXIS; ++i) {
        ans[i] = v[i] / length;
    }
    return ans;
}

__host__ __device__ Vec3 prod (Vec3 a, Vec3 b) {
    return createVec3 (a[Y] * b[Z] - a[Z] * b[Y],
            a[Z] * b[X] - a[X] * b[Z],
            a[X] * b[Y] - a[Y] * b[X]);
}

__host__ __device__ Vec3 multiple (Vec3 a, Vec3 b, Vec3 c, Vec3 d) {
    return createVec3 (a[X] * d[X] + b[X] * d[Y] + c[X] * d[Z],
            a[Y] * d[X] + b[Y] * d[Y] + c[Y] * d[Z],
            a[Z] * d[X] + b[Z] * d[Y] + c[Z] * d[Z]);
}

__host__ __device__ Vec3 reflect (Vec3 vec, Vec3 normal) {
    double dot_mult = dot(vec, normal) * (-2.0);
    Vec3 part = normal * dot_mult;
    return vec + part;
}