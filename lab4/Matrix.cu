#include "Matrix.cuh"

__global__ void ReverseKernel (uint64_t n, double *data) {
    
}

Matrix::Matrix () : n(0), m(0) {}

Matrix::Matrix (Matrix const &matrix) {
    n = matrix.n;
    m = matrix.m;
    data = matrix.data;
}

Matrix::Matrix (Matrix &&matrix) {
    n = matrix.n;
    m = matrix.m;
    data = std::move(matrix.data);
}

Matrix::Matrix (uint32_t n, uint32_t m, const std::vector<double> &buff) : n(n), m(m), data(buff) {}

Matrix::~Matrix () {}

double &Matrix::operator() (uint64_t i, uint64_t j) {
    return data[i * n + j];
}

double Matrix::operator() (uint64_t i, uint64_t j) const {
    return data[i * n + j];
}

// Matrix &Matrix::operator+= (const Matrix &matrix) {
//     for (uint64_t i = 0; i < data.size(); ++i) {
//         data[i] += matrix.data[i];
//     }
//     return *this;
// }

// Matrix &Matrix::operator-= (const Matrix &matrix) {
//     for (uint64_t i = 0; i < 9; ++i) {
//         data[i] -= matrix.data[i];
//     }
//     return *this;
// }

// Matrix &Matrix::operator*= (double el) {
//     for (uint64_t i = 0; i < 9; ++i) {
//         data[i] *= el;
//     }
//     return *this;
// }

// Matrix &Matrix::operator/= (double el) {
//     for (uint64_t i = 0; i < 9; ++i) {
//         data[i] /= el;
//     }
//     return *this;
// }

Matrix &Matrix::operator= (const Matrix &matrix) {
    n = matrix.n;
    m = matrix.m;
    data = matrix.data;
    return *this;
}

Matrix &Matrix::operator= (Matrix &&matrix) {
    n = matrix.n;
    m = matrix.m;
    data = std::move(matrix.data);
    return *this;
}

Matrix Matrix::reverse () const {
    Matrix ans;
    return ans;
}

// const Matrix operator+ (const Matrix &m1, const Matrix &m2) {
//     Matrix ans;
//     for (uint64_t i = 0; i < 9; ++i) {
//         ans.data[i] = m1.data[i] + m2.data[i];
//     }
//     return ans;
// }

// const Matrix operator- (const Matrix &m1, const Matrix &m2) {
//     Matrix ans;
//     for (uint64_t i = 0; i < 9; ++i) {
//         ans.data[i] = m1.data[i] - m2.data[i];
//     }
//     return ans;
// }

// const Matrix operator* (const Matrix &matrix, double el) {
//     Matrix ans = matrix;
//     for (uint64_t i = 0; i < 9; ++i) {
//         ans.data[i] *= el;
//     }
//     return ans;
// }

// const Matrix operator/ (const Matrix &matrix, double el) {
//     Matrix ans = matrix;
//     for (uint64_t i = 0; i < 9; ++i) {
//         ans.data[i] /= el;
//     }
//     return ans;
// }

std::istream &operator>> (std::istream &input, Matrix &matrix) {
    for (uint64_t i = 0; i < matrix.data.size(); ++i) {
        input >> matrix.data[i];
    }
    return input;
}

std::ostream &operator<< (std::ostream &output, Matrix &matrix) {
    for (uint64_t i = 0; i < matrix.n; ++i) {
        for (uint64_t j = 0; j < matrix.m - 1; ++j) {
            output << matrix.data[i * matrix.m + j] << ' ';
        }
        output << matrix.data[i * matrix.m + matrix.m - 1] << '\n';
    }
    return output;
}