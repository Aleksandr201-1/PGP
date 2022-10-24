#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <iostream>
#include <cstdint>
#include <vector>

class Matrix {
    public:
        Matrix ();
        Matrix (Matrix const &matrix);
        Matrix (Matrix &&matrix);
        Matrix (uint32_t w, uint32_t h, const std::vector<double> &buff);
        ~Matrix ();

        double &operator() (uint64_t i, uint64_t j);
        double operator() (uint64_t i, uint64_t j) const;

        // Matrix &operator+= (const Matrix &matrix);
        // Matrix &operator-= (const Matrix &matrix);
        // Matrix &operator*= (double el);
        // Matrix &operator/= (double el);

        Matrix &operator= (const Matrix &matrix);
        Matrix &operator= (Matrix &&matrix);

        Matrix reverse () const;

        // friend const Matrix operator+ (const Matrix &m1, const Matrix &m2);
        // friend const Matrix operator- (const Matrix &m1, const Matrix &m2);
        // friend const Matrix operator* (const Matrix &m1, double el);
        // friend const Matrix operator/ (const Matrix &m1, double el);

        friend std::istream &operator>> (std::istream &input, Matrix &matrix);
        friend std::ostream &operator<< (std::ostream &output, Matrix &matrix);
    private:
        uint64_t n, m;
        std::vector<double> data;
};

#endif