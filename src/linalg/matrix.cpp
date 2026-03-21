#include "ml/linalg/matrix.hpp"
#include "ml/linalg/vector.hpp"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace {
    constexpr double EPS = 1e-12;

    inline void swap_rows(std::vector<double>& data, size_t cols, size_t r1, size_t r2) {
        if (r1 == r2) return;
        for (size_t c = 0; c < cols; ++c) {
            std::swap(data[r1 * cols + c], data[r2 * cols + c]);
        }
    }
}

// ================================
// Helpers
// ================================

size_t Matrix::index(size_t row, size_t col) const {
    return row * cols_ + col;
}

Matrix Matrix::minorMatrix(size_t row, size_t col) const {
    if (rows_ == 0 || cols_ == 0 || rows_ == 1 || cols_ == 1) {
        return Matrix(0, 0);
    }

    Matrix result(rows_ - 1, cols_ - 1);
    size_t r = 0;

    for (size_t i = 0; i < rows_; ++i) {
        if (i == row) continue;

        size_t c = 0;
        for (size_t j = 0; j < cols_; ++j) {
            if (j == col) continue;
            result(r, c++) = (*this)(i, j);
        }
        ++r;
    }

    return result;
}

// ================================
// Constructors
// ================================

Matrix::Matrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), data(rows * cols, 0.0) {}

Matrix::Matrix(size_t rows, size_t cols, double initialValue)
    : rows_(rows), cols_(cols), data(rows * cols, initialValue) {}

// ================================
// Move Semantics
// ================================

Matrix::Matrix(Matrix&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), data(std::move(other.data)) {
    other.rows_ = 0;
    other.cols_ = 0;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        data = std::move(other.data);

        other.rows_ = 0;
        other.cols_ = 0;
    }
    return *this;
}

// ================================
// Copy Semantics
// ================================

Matrix::Matrix(const Matrix& other)
    : rows_(other.rows_), cols_(other.cols_), data(other.data) {}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        data = other.data;
    }
    return *this;
}

// ================================
// Shape
// ================================

size_t Matrix::rows() const {
    return rows_;
}

size_t Matrix::cols() const {
    return cols_;
}

// ================================
// Element Access
// ================================

double Matrix::get(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data[index(row, col)];
}

void Matrix::set(size_t row, size_t col, double value) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
    data[index(row, col)] = value;
}

double& Matrix::operator()(size_t row, size_t col) {
    return data[index(row, col)];
}

const double& Matrix::operator()(size_t row, size_t col) const {
    return data[index(row, col)];
}

// ================================
// Matrix Operations
// ================================

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix addition shape mismatch");
    }

    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix subtraction shape mismatch");
    }

    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

// ================================
// In-place operations
// ================================

Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix += shape mismatch");
    }

    for (size_t i = 0; i < data.size(); ++i) {
        data[i] += other.data[i];
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix -= shape mismatch");
    }

    for (size_t i = 0; i < data.size(); ++i) {
        data[i] -= other.data[i];
    }
    return *this;
}

Matrix& Matrix::operator*=(double scalar) {
    for (double& v : data) {
        v *= scalar;
    }
    return *this;
}

Matrix& Matrix::operator/=(double scalar) {
    if (std::abs(scalar) < EPS) {
        throw std::invalid_argument("Division by zero");
    }

    for (double& v : data) {
        v /= scalar;
    }
    return *this;
}

// ================================
// Matrix-Vector Multiplication
// ================================

Vector Matrix::operator*(const Vector& vec) const {
    if (cols_ != vec.size()) {
        throw std::invalid_argument("Matrix-vector multiplication shape mismatch");
    }

    Vector result(rows_, 0.0);

    for (size_t i = 0; i < rows_; ++i) {
        double sum = 0.0;
        size_t base = i * cols_;
        for (size_t j = 0; j < cols_; ++j) {
            sum += data[base + j] * vec[j];
        }
        result[i] = sum;
    }

    return result;
}

Matrix operator*(double scalar, const Matrix& mat) {
    return mat * scalar;
}

// ================================
// Matrix-Matrix Multiplication
// ================================

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix multiplication shape mismatch");
    }

    Matrix result(rows_, other.cols_, 0.0);

    for (size_t i = 0; i < rows_; ++i) {
        for (size_t k = 0; k < cols_; ++k) {
            const double aik = data[i * cols_ + k];
            if (std::abs(aik) < EPS) continue;

            size_t lhsBase = i * other.cols_;
            size_t rhsBase = k * other.cols_;

            for (size_t j = 0; j < other.cols_; ++j) {
                result.data[lhsBase + j] += aik * other.data[rhsBase + j];
            }
        }
    }

    return result;
}

// ================================
// Linear Algebra
// ================================

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

double Matrix::determinant() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("Determinant requires a square matrix");
    }

    if (rows_ == 0) {
        throw std::invalid_argument("Determinant of empty matrix is undefined");
    }

    if (rows_ == 1) {
        return data[0];
    }

    std::vector<double> a = data;
    double det = 1.0;
    int sign = 1;

    for (size_t i = 0; i < rows_; ++i) {
        // Pivot selection
        size_t pivot = i;
        double maxAbs = std::abs(a[i * cols_ + i]);

        for (size_t r = i + 1; r < rows_; ++r) {
            double val = std::abs(a[r * cols_ + i]);
            if (val > maxAbs) {
                maxAbs = val;
                pivot = r;
            }
        }

        if (maxAbs < EPS) {
            return 0.0;
        }

        if (pivot != i) {
            swap_rows(a, cols_, i, pivot);
            sign = -sign;
        }

        double pivotVal = a[i * cols_ + i];
        det *= pivotVal;

        // Eliminate below
        for (size_t r = i + 1; r < rows_; ++r) {
            double factor = a[r * cols_ + i] / pivotVal;
            if (std::abs(factor) < EPS) continue;

            a[r * cols_ + i] = 0.0;
            for (size_t c = i + 1; c < cols_; ++c) {
                a[r * cols_ + c] -= factor * a[i * cols_ + c];
            }
        }
    }

    return det * sign;
}

Matrix Matrix::inverse() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("Inverse requires a square matrix");
    }

    if (rows_ == 0) {
        throw std::invalid_argument("Inverse of empty matrix is undefined");
    }

    const size_t n = rows_;
    std::vector<double> a = data;
    Matrix inv = Matrix::identity(n);

    for (size_t i = 0; i < n; ++i) {
        // Pivot selection
        size_t pivot = i;
        double maxAbs = std::abs(a[i * n + i]);

        for (size_t r = i + 1; r < n; ++r) {
            double val = std::abs(a[r * n + i]);
            if (val > maxAbs) {
                maxAbs = val;
                pivot = r;
            }
        }

        if (maxAbs < EPS) {
            throw std::runtime_error("Matrix is singular and cannot be inverted");
        }

        if (pivot != i) {
            swap_rows(a, n, i, pivot);
            swap_rows(inv.data, n, i, pivot);
        }

        // Normalize pivot row
        double pivotVal = a[i * n + i];
        for (size_t c = 0; c < n; ++c) {
            a[i * n + c] /= pivotVal;
            inv(i, c) /= pivotVal;
        }

        // Eliminate other rows
        for (size_t r = 0; r < n; ++r) {
            if (r == i) continue;

            double factor = a[r * n + i];
            if (std::abs(factor) < EPS) continue;

            a[r * n + i] = 0.0;
            for (size_t c = 0; c < n; ++c) {
                a[r * n + c] -= factor * a[i * n + c];
                inv(r, c) -= factor * inv(i, c);
            }
        }
    }

    return inv;
}

// ================================
// Row / Column Access
// ================================

Vector Matrix::row(size_t r) const {
    if (r >= rows_) {
        throw std::out_of_range("Row index out of range");
    }

    Vector result(cols_, 0.0);
    size_t base = r * cols_;
    for (size_t j = 0; j < cols_; ++j) {
        result[j] = data[base + j];
    }
    return result;
}

Vector Matrix::col(size_t c) const {
    if (c >= cols_) {
        throw std::out_of_range("Column index out of range");
    }

    Vector result(rows_, 0.0);
    for (size_t i = 0; i < rows_; ++i) {
        result[i] = (*this)(i, c);
    }
    return result;
}

// ================================
// Utilities
// ================================

void Matrix::print() const {
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            std::cout << (*this)(i, j) << " ";
        }
        std::cout << '\n';
    }
}

Matrix Matrix::identity(size_t n) {
    Matrix id(n, n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        id(i, i) = 1.0;
    }
    return id;
}

double Matrix::sum() const {
    double s = 0.0;
    for (double v : data) {
        s += v;
    }
    return s;
}

double Matrix::mean() const {
    if (data.empty()) {
        throw std::invalid_argument("Mean of empty matrix is undefined");
    }
    return sum() / data.size();
}

Matrix Matrix::hadamard(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Hadamard shape mismatch");
    }

    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

Matrix Matrix::normalize() const {
    if (data.empty()) {
        return Matrix(rows_, cols_);
    }

    double min_val = data[0];
    double max_val = data[0];

    for (double v : data) {
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }

    double range = max_val - min_val;
    Matrix result(rows_, cols_, 0.0);

    if (std::abs(range) < EPS) {
        return result;
    }

    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = (data[i] - min_val) / range;
    }

    return result;
}

// ================================
// Statistics
// ================================

double Matrix::trace() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("Trace requires a square matrix");
    }

    double t = 0.0;
    for (size_t i = 0; i < rows_; ++i) {
        t += (*this)(i, i);
    }
    return t;
}

Vector Matrix::columnMean() const {
    if (rows_ == 0) {
        return Vector(cols_, 0.0);
    }

    Vector result(cols_, 0.0);
    for (size_t j = 0; j < cols_; ++j) {
        double s = 0.0;
        for (size_t i = 0; i < rows_; ++i) {
            s += (*this)(i, j);
        }
        result[j] = s / rows_;
    }
    return result;
}

Matrix Matrix::standardize() const {
    if (data.empty()) {
        return Matrix(rows_, cols_);
    }

    double m = mean();

    double var = 0.0;
    for (double v : data) {
        double d = v - m;
        var += d * d;
    }
    var /= data.size();

    double stddev = std::sqrt(var);
    Matrix result(rows_, cols_, 0.0);

    if (std::abs(stddev) < EPS) {
        return result;
    }

    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = (data[i] - m) / stddev;
    }

    return result;
}

Matrix Matrix::covariance() const {
    // Assumes rows = samples, cols = features
    if (rows_ == 0 || cols_ == 0) {
        return Matrix(cols_, cols_, 0.0);
    }

    if (rows_ < 2) {
        return Matrix(cols_, cols_, 0.0);
    }

    Vector means = columnMean();
    Matrix cov(cols_, cols_, 0.0);

    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            double xj = (*this)(i, j) - means[j];
            for (size_t k = j; k < cols_; ++k) {
                double xk = (*this)(i, k) - means[k];
                cov(j, k) += xj * xk;
            }
        }
    }

    double scale = 1.0 / (rows_ - 1);
    for (size_t j = 0; j < cols_; ++j) {
        for (size_t k = j; k < cols_; ++k) {
            cov(j, k) *= scale;
            if (j != k) {
                cov(k, j) = cov(j, k);
            }
        }
    }

    return cov;
}

Matrix Matrix::correlation() const {
    Matrix cov = covariance();
    Matrix result(cols_, cols_, 0.0);

    for (size_t i = 0; i < cols_; ++i) {
        double diag_i = cov(i, i);
        double std_i = diag_i > 0.0 ? std::sqrt(diag_i) : 0.0;

        for (size_t j = i; j < cols_; ++j) {
            double diag_j = cov(j, j);
            double std_j = diag_j > 0.0 ? std::sqrt(diag_j) : 0.0;

            double value = 0.0;
            if (std_i > EPS && std_j > EPS) {
                value = cov(i, j) / (std_i * std_j);
            }

            result(i, j) = value;
            result(j, i) = value;
        }
    }

    return result;
}

// ================================
// Scalar Ops
// ================================

Matrix Matrix::operator+(double scalar) const {
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + scalar;
    }
    return result;
}

Matrix Matrix::operator-(double scalar) const {
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] - scalar;
    }
    return result;
}

Matrix Matrix::operator/(double scalar) const {
    if (std::abs(scalar) < EPS) {
        throw std::invalid_argument("Division by zero");
    }

    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] / scalar;
    }
    return result;
}

// ================================
// Axis Ops
// ================================

Vector Matrix::sum(int axis) const {
    if (axis == 0) {
        Vector result(cols_, 0.0);
        for (size_t j = 0; j < cols_; ++j) {
            double s = 0.0;
            for (size_t i = 0; i < rows_; ++i) {
                s += (*this)(i, j);
            }
            result[j] = s;
        }
        return result;
    }

    if (axis == 1) {
        Vector result(rows_, 0.0);
        for (size_t i = 0; i < rows_; ++i) {
            double s = 0.0;
            size_t base = i * cols_;
            for (size_t j = 0; j < cols_; ++j) {
                s += data[base + j];
            }
            result[i] = s;
        }
        return result;
    }

    throw std::invalid_argument("Axis must be 0 or 1");
}

Vector Matrix::mean(int axis) const {
    if (axis == 0) {
        Vector result(cols_, 0.0);
        for (size_t j = 0; j < cols_; ++j) {
            double s = 0.0;
            for (size_t i = 0; i < rows_; ++i) {
                s += (*this)(i, j);
            }
            result[j] = s / rows_;
        }
        return result;
    }

    if (axis == 1) {
        Vector result(rows_, 0.0);
        for (size_t i = 0; i < rows_; ++i) {
            double s = 0.0;
            size_t base = i * cols_;
            for (size_t j = 0; j < cols_; ++j) {
                s += data[base + j];
            }
            result[i] = s / cols_;
        }
        return result;
    }

    throw std::invalid_argument("Axis must be 0 or 1");
}