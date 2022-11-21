#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <cstdint>

struct Matrix {
    double data[9];
    __host__ __device__ double &operator() (uint64_t i, uint64_t j);
    __host__ __device__ double operator() (uint64_t i, uint64_t j) const;

    __host__ __device__ Matrix &operator+= (const Matrix &m);
    __host__ __device__ Matrix &operator-= (const Matrix &m);
    __host__ __device__ Matrix &operator*= (double el);
    __host__ __device__ Matrix &operator/= (double el);

    friend const Matrix operator+ (const Matrix &m1, const Matrix &m2);
    friend const Matrix operator- (const Matrix &m1, const Matrix &m2);
    friend const Matrix operator* (const Matrix &m1, double el);
    friend const Matrix operator/ (const Matrix &m1, double el);

    __host__ __device__ Matrix &operator= (const Matrix &m);
};

__host__ __device__ void initMatrix (Matrix &m);

void reverseMatrix (Matrix &m);

#endif