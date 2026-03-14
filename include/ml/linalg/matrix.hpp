#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <cstddef>
#include <stdexcept>
#include "ml/linalg/vector.hpp"

class Matrix {
private:
    size_t rows_; // number of rows
    size_t cols_; // number of columns
    std::vector<double> data; // flattened matrix storage (row-major)

    size_t index(size_t row, size_t col) const; // convert (row,col) to flat index
    Matrix minorMatrix(size_t row, size_t col) const; // compute minor matrix

public:

    /* Constructors */

    explicit Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, double initialValue);

    /* Shape */

    [[nodiscard]] size_t rows() const;
    [[nodiscard]] size_t cols() const;

    /* Element Access */

    double get(size_t row, size_t col) const;
    void set(size_t row, size_t col, double value);

    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;

    /* Matrix Operations */

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(double scalar) const;

    /* Matrix-Vector Multiplication */

    Vector operator*(const Vector& vec) const;
    friend Matrix operator*(double scalar, const Matrix& mat);

    /* Matrix-Matrix Multiplication */

    Matrix operator*(const Matrix& other) const;

    /* Linear Algebra */

    [[nodiscard]] Matrix transpose() const;
    [[nodiscard]] double determinant() const;
    [[nodiscard]] Matrix inverse() const;

    /* Row / Column Access */

    Vector row(size_t r) const;
    Vector col(size_t c) const;

    /* Utilities */

    void print() const;

    static Matrix identity(size_t n);

    double sum() const;
    double mean() const;

    Matrix hadamard(const Matrix& other) const;
    Matrix normalize() const;

    /* Statistics */

    double trace() const;

    Vector columnMean() const;

    Matrix standardize() const;

    Matrix covariance() const;

    Matrix correlation() const;
};

#endif