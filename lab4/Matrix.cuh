#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <iostream>
#include <cstdint>
#include <vector>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include "GPUErrorCheck.cuh"

class Matrix {
    public:
        Matrix ();
        Matrix (uint64_t size);
        Matrix (Matrix const &matrix);
        Matrix (Matrix &&matrix);
        Matrix (uint64_t n, uint64_t m, const std::vector<double> &buff);
        Matrix (uint64_t n, uint64_t m, double *buff);
        Matrix (uint64_t n, uint64_t m);
        ~Matrix ();

        double &operator() (uint64_t i, uint64_t j);
        double operator() (uint64_t i, uint64_t j) const;

        Matrix &operator= (const Matrix &matrix);
        Matrix &operator= (Matrix &&matrix);

        Matrix reverse () const;

        friend const Matrix operator* (const Matrix &m1, const Matrix &m2);

        friend std::istream &operator>> (std::istream &input, Matrix &matrix);
        friend std::ostream &operator<< (std::ostream &output, const Matrix &matrix);
    private:
        uint64_t n, m;
        std::vector<double> data;
};

#endif