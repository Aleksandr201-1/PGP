#include "Matrix.cuh"

__host__ __device__ double &Matrix::operator() (uint64_t i, uint64_t j) {
    return data[i * 3 + j];
}

__host__ __device__ double Matrix::operator() (uint64_t i, uint64_t j) const {
    return data[i * 3 + j];
}

__host__ __device__ Matrix &Matrix::operator+= (const Matrix &m) {
    for (uint64_t i = 0; i < 9; ++i) {
        data[i] += m.data[i];
    }
    return *this;
}

__host__ __device__ Matrix &Matrix::operator-= (const Matrix &m) {
    for (uint64_t i = 0; i < 9; ++i) {
        data[i] -= m.data[i];
    }
    return *this;
}

__host__ __device__ Matrix &Matrix::operator*= (double el) {
    for (uint64_t i = 0; i < 9; ++i) {
        data[i] *= el;
    }
    return *this;
}

__host__ __device__ Matrix &Matrix::operator/= (double el) {
    for (uint64_t i = 0; i < 9; ++i) {
        data[i] /= el;
    }
    return *this;
}

const Matrix operator+ (const Matrix &m1, const Matrix &m2) {
    Matrix ans;
    for (uint64_t i = 0; i < 9; ++i) {
        ans.data[i] = m1.data[i] + m2.data[i];
    }
    return ans;
}

const Matrix operator- (const Matrix &m1, const Matrix &m2) {
    Matrix ans;
    for (uint64_t i = 0; i < 9; ++i) {
        ans.data[i] = m1.data[i] - m2.data[i];
    }
    return ans;
}

const Matrix operator* (const Matrix &m1, double el) {
    Matrix ans = m1;
    for (uint64_t i = 0; i < 9; ++i) {
        ans.data[i] *= el;
    }
    return ans;
}

const Matrix operator/ (const Matrix &m1, double el) {
    Matrix ans = m1;
    for (uint64_t i = 0; i < 9; ++i) {
        ans.data[i] /= el;
    }
    return ans;
}

__host__ __device__ Matrix &Matrix::operator= (const Matrix &m) {
    for (uint64_t i = 0; i < 3; ++i) {
        for (uint64_t j = 0; j < 3; ++j) {
            data[i * 3 + j] = m(i, j);
        }
    }
    return *this;
}

__host__ __device__ void initMatrix (Matrix &m) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            m(j, i) = 0;
        }
    }
}

void reverseMatrix (Matrix &m) {
    Matrix tmp;
    tmp(0, 0) = (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2));
    tmp(0, 1) = -(m(1, 0) * m(2, 2) - m(2, 0) * m(1, 2));
    tmp(0, 2) = (m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1));

    tmp(1, 0) = -(m(0, 1) * m(2, 2) - m(2, 1) * m(0, 2));
    tmp(1, 1) = (m(0, 0) * m(2, 2) - m(2, 0) * m(0, 2));
    tmp(1, 2) = -(m(0, 0) * m(2, 1) - m(2, 0) * m(0, 1));

    tmp(2, 0) = (m(0, 1) * m(1, 2) - m(1, 1) * m(0, 2));
    tmp(2, 1) = -(m(0, 0) * m(1, 2) - m(1, 0) * m(0, 2));
    tmp(2, 2) = (m(0, 0) * m(1, 1) - m(1, 0) * m(0, 1));

    double det = m(0, 0) * tmp(0, 0) + m(0, 1) * tmp(0, 1) + m(0, 2) * tmp(0, 2);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            tmp(i, j) /= det;
            m(j, i) = tmp(i, j);
        }
    }
}