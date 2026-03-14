#include "../include/matrix.hpp"

#include <stdexcept>
#include <iostream>


/* Private Index Helper */

size_t Matrix::index(size_t row, size_t col) const {

    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }

    return row * cols_ + col;
}


/* Constructors */

Matrix::Matrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), data(rows * cols, 0.0) {}

Matrix::Matrix(size_t rows, size_t cols, double initialValue)
    : rows_(rows), cols_(cols), data(rows * cols, initialValue) {}


/* Shape */

size_t Matrix::rows() const {
    return rows_;
}

size_t Matrix::cols() const {
    return cols_;
}


/* Element Access */

double Matrix::get(size_t row, size_t col) const {
    return data[index(row, col)];
}

void Matrix::set(size_t row, size_t col, double value) {
    data[index(row, col)] = value;
}

double& Matrix::operator()(size_t row, size_t col) {
    return data[index(row, col)];
}

const double& Matrix::operator()(size_t row, size_t col) const {
    return data[index(row, col)];
}


/* Matrix Addition */

Matrix Matrix::operator+(const Matrix& other) const {

    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix sizes must match for addition");
    }

    Matrix result(rows_, cols_);

    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = data[i] + other.data[i];
    }

    return result;
}


/* Matrix Subtraction */

Matrix Matrix::operator-(const Matrix& other) const {

    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix sizes must match for subtraction");
    }

    Matrix result(rows_, cols_);

    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = data[i] - other.data[i];
    }

    return result;
}


/* Scalar Multiplication */

Matrix Matrix::operator*(double scalar) const {

    Matrix result(rows_, cols_);

    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = data[i] * scalar;
    }

    return result;
}


/* Matrix-Vector Multiplication */

Vector Matrix::operator*(const Vector& vec) const {

    if (cols_ != vec.size()) {
        throw std::invalid_argument("Matrix columns must match vector size");
    }

    Vector result(rows_);

    for (size_t i = 0; i < rows_; i++) {

        double sum = 0.0;

        for (size_t j = 0; j < cols_; j++) {
            sum += (*this)(i, j) * vec[j];
        }

        result[i] = sum;
    }

    return result;
}


/* Matrix-Matrix Multiplication */

Matrix Matrix::operator*(const Matrix& other) const {

    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix dimensions invalid for multiplication");
    }

    Matrix result(rows_, other.cols_);

    for (size_t i = 0; i < rows_; i++) {

        for (size_t j = 0; j < other.cols_; j++) {

            double sum = 0.0;

            for (size_t k = 0; k < cols_; k++) {
                sum += (*this)(i, k) * other(k, j);
            }

            result(i, j) = sum;
        }
    }

    return result;
}


/* Transpose */

Matrix Matrix::transpose() const {

    Matrix result(cols_, rows_);

    for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < cols_; j++) {
            result(j, i) = (*this)(i, j);
        }
    }

    return result;
}


/* Row Extraction */

Vector Matrix::row(size_t r) const {

    if (r >= rows_) {
        throw std::out_of_range("Row index out of range");
    }

    Vector result(cols_);

    for (size_t j = 0; j < cols_; j++) {
        result[j] = (*this)(r, j);
    }

    return result;
}


/* Column Extraction */

Vector Matrix::col(size_t c) const {

    if (c >= cols_) {
        throw std::out_of_range("Column index out of range");
    }

    Vector result(rows_);

    for (size_t i = 0; i < rows_; i++) {
        result[i] = (*this)(i, c);
    }

    return result;
}


/* Print */

void Matrix::print() const {

    for (size_t i = 0; i < rows_; i++) {

        std::cout << "[ ";

        for (size_t j = 0; j < cols_; j++) {
            std::cout << (*this)(i, j) << " ";
        }

        std::cout << "]" << std::endl;
    }
}