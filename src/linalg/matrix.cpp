#include "ml/linalg/matrix.hpp"
#include "ml/linalg/vector.hpp"

#include <stdexcept>
#include <iostream>
#include <cmath>

/* Move Semantics */

Matrix::Matrix(Matrix&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), data(std::move(other.data)) {}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        data = std::move(other.data);
    }
    return *this;
}


/* Private Index Helper */

size_t Matrix::index(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_)
        throw std::out_of_range("Matrix index out of range");

    return row * cols_ + col;
}


/* Constructors */

Matrix::Matrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), data(rows * cols, 0.0) {}

Matrix::Matrix(size_t rows, size_t cols, double initialValue)
    : rows_(rows), cols_(cols), data(rows * cols, initialValue) {}


/* Shape */

size_t Matrix::rows() const { return rows_; }
size_t Matrix::cols() const { return cols_; }


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
    if (rows_ != other.rows_ || cols_ != other.cols_)
        throw std::invalid_argument("Matrix sizes must match for addition");

    Matrix result(rows_, cols_);
    size_t n = data.size();

    for (size_t i = 0; i < n; i++)
        result.data[i] = data[i] + other.data[i];

    return result;
}


/* Matrix Subtraction */

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_)
        throw std::invalid_argument("Matrix sizes must match for subtraction");

    Matrix result(rows_, cols_);
    size_t n = data.size();

    for (size_t i = 0; i < n; i++)
        result.data[i] = data[i] - other.data[i];

    return result;
}


/* Scalar Multiplication */

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows_, cols_);
    size_t n = data.size();

    for (size_t i = 0; i < n; i++)
        result.data[i] = data[i] * scalar;

    return result;
}


/* In-place Operations */

Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_)
        throw std::invalid_argument("Size mismatch");

    for (size_t i = 0; i < data.size(); ++i)
        data[i] += other.data[i];

    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_)
        throw std::invalid_argument("Size mismatch");

    for (size_t i = 0; i < data.size(); ++i)
        data[i] -= other.data[i];

    return *this;
}

Matrix& Matrix::operator*=(double scalar) {
    for (double& val : data)
        val *= scalar;

    return *this;
}

Matrix& Matrix::operator/=(double scalar) {
    if (scalar == 0.0)
        throw std::invalid_argument("Division by zero");

    for (double& val : data)
        val /= scalar;

    return *this;
}


/* Matrix-Vector Multiplication */

Vector Matrix::operator*(const Vector& vec) const {
    if (cols_ != vec.size())
        throw std::invalid_argument("Matrix columns must match vector size");

    Vector result(rows_);

    for (size_t i = 0; i < rows_; i++) {
        double sum = 0.0;

        for (size_t j = 0; j < cols_; j++)
            sum += data[i * cols_ + j] * vec[j];

        result[i] = sum;
    }

    return result;
}


/* Matrix-Matrix Multiplication */

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_)
        throw std::invalid_argument("Matrix dimensions invalid for multiplication");

    Matrix result(rows_, other.cols_);

    for (size_t i = 0; i < rows_; i++) {
        for (size_t k = 0; k < cols_; k++) {
            double val = data[i * cols_ + k];

            for (size_t j = 0; j < other.cols_; j++) {
                result.data[i * other.cols_ + j] += val * other.data[k * other.cols_ + j];
            }
        }
    }

    return result;
}


/* Transpose */

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);

    for (size_t i = 0; i < rows_; i++)
        for (size_t j = 0; j < cols_; j++)
            result.data[j * rows_ + i] = data[i * cols_ + j];

    return result;
}


/* Row / Column */

Vector Matrix::row(size_t r) const {
    if (r >= rows_)
        throw std::out_of_range("Row index out of range");

    Vector result(cols_);

    for (size_t j = 0; j < cols_; j++)
        result[j] = data[r * cols_ + j];

    return result;
}

Vector Matrix::col(size_t c) const {
    if (c >= cols_)
        throw std::out_of_range("Column index out of range");

    Vector result(rows_);

    for (size_t i = 0; i < rows_; i++)
        result[i] = data[i * cols_ + c];

    return result;
}


/* Print */

void Matrix::print() const {
    for (size_t i = 0; i < rows_; i++) {
        std::cout << "[ ";
        for (size_t j = 0; j < cols_; j++)
            std::cout << data[i * cols_ + j] << " ";
        std::cout << "]\n";
    }
}


/* Identity */

Matrix Matrix::identity(size_t n) {
    Matrix I(n, n);

    for (size_t i = 0; i < n; i++)
        I.data[i * n + i] = 1.0;

    return I;
}


/* Sum / Mean */

double Matrix::sum() const {
    double s = 0.0;
    for (double v : data) s += v;
    return s;
}

double Matrix::mean() const {
    if (data.empty())
        throw std::runtime_error("Cannot compute mean of empty matrix");

    return sum() / data.size();
}


/* Hadamard */

Matrix Matrix::hadamard(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_)
        throw std::invalid_argument("Matrix dimensions must match");

    Matrix result(rows_, cols_);

    for (size_t i = 0; i < data.size(); i++)
        result.data[i] = data[i] * other.data[i];

    return result;
}


/* Apply */

Matrix Matrix::apply(std::function<double(double)> func) const {
    Matrix result(rows_, cols_);

    for (size_t i = 0; i < data.size(); ++i)
        result.data[i] = func(data[i]);

    return result;
}


/* Scalar Ops */

Matrix Matrix::operator+(double scalar) const {
    Matrix result(rows_, cols_);

    for (size_t i = 0; i < data.size(); ++i)
        result.data[i] = data[i] + scalar;

    return result;
}

Matrix Matrix::operator-(double scalar) const {
    Matrix result(rows_, cols_);

    for (size_t i = 0; i < data.size(); ++i)
        result.data[i] = data[i] - scalar;

    return result;
}

Matrix Matrix::operator/(double scalar) const {
    if (scalar == 0.0)
        throw std::invalid_argument("Division by zero");

    Matrix result(rows_, cols_);

    for (size_t i = 0; i < data.size(); ++i)
        result.data[i] = data[i] / scalar;

    return result;
}


/* Axis Ops */

Vector Matrix::sum(int axis) const {
    if (axis == 0) {
        Vector result(cols_, 0.0);

        for (size_t j = 0; j < cols_; ++j)
            for (size_t i = 0; i < rows_; ++i)
                result[j] += data[i * cols_ + j];

        return result;
    }
    else if (axis == 1) {
        Vector result(rows_, 0.0);

        for (size_t i = 0; i < rows_; ++i)
            for (size_t j = 0; j < cols_; ++j)
                result[i] += data[i * cols_ + j];

        return result;
    }
    else {
        throw std::invalid_argument("Axis must be 0 or 1");
    }
}

Vector Matrix::mean(int axis) const {
    if (axis == 0)
        return sum(0) / static_cast<double>(rows_);
    else if (axis == 1)
        return sum(1) / static_cast<double>(cols_);
    else
        throw std::invalid_argument("Axis must be 0 or 1");
}