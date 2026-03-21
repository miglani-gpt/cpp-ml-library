#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <cstddef>
#include <stdexcept>

class Vector; // forward declaration

class Matrix {
private:
    size_t rows_;
    size_t cols_;
    std::vector<double> data;

    size_t index(size_t row, size_t col) const;
    Matrix minorMatrix(size_t row, size_t col) const;

public:

    /* Constructors */
    explicit Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, double initialValue);

    /* Move Semantics */
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;

    // Copy semantics
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);

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

    /* In-place operations */
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(double scalar);
    Matrix& operator/=(double scalar);

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

    /* ML Ops */
    template <typename Func>
    Matrix apply(Func f) const {
        Matrix result(rows(), cols());   // ✅ correct
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result(i, j) = f((*this)(i, j));
            }
        }
        return result;
    }

    /* Scalar Ops */
    Matrix operator+(double scalar) const;
    Matrix operator-(double scalar) const;
    Matrix operator/(double scalar) const;

    /* Axis Ops */
    Vector sum(int axis) const;
    Vector mean(int axis) const;
};

#endif