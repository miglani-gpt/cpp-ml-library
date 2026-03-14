#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <cstddef>
#include "vector.hpp"

class Matrix {
private:
    size_t rows_;
    size_t cols_;
    std::vector<double> data;

    size_t index(size_t row, size_t col) const;

public:

    /* Constructors */

    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, double initialValue);


    /* Shape */

    size_t rows() const;
    size_t cols() const;


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


    /* Matrix-Matrix Multiplication */

    Matrix operator*(const Matrix& other) const;


    /* Linear Algebra */

    Matrix transpose() const;


    /* Row / Column Access */

    Vector row(size_t r) const;
    Vector col(size_t c) const;


    /* Utilities */

    void print() const;
};

#endif