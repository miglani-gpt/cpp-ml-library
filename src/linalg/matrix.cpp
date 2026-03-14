#include "ml/linalg/matrix.hpp"

#include <stdexcept>
#include <iostream>
#include <cmath>

/* Private Index Helper */

size_t Matrix::index(size_t row, size_t col) const
{
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

double Matrix::get(size_t row, size_t col) const
{
    return data[index(row, col)];
}

void Matrix::set(size_t row, size_t col, double value)
{
    data[index(row, col)] = value;
}

double& Matrix::operator()(size_t row, size_t col)
{
    return data[index(row, col)];
}

const double& Matrix::operator()(size_t row, size_t col) const
{
    return data[index(row, col)];
}


/* Matrix Addition */

Matrix Matrix::operator+(const Matrix& other) const
{
    if (rows_ != other.rows_ || cols_ != other.cols_)
        throw std::invalid_argument("Matrix sizes must match for addition");

    Matrix result(rows_, cols_);

    size_t n = data.size();

    for (size_t i = 0; i < n; i++)
        result.data[i] = data[i] + other.data[i];

    return result;
}


/* Matrix Subtraction */

Matrix Matrix::operator-(const Matrix& other) const
{
    if (rows_ != other.rows_ || cols_ != other.cols_)
        throw std::invalid_argument("Matrix sizes must match for subtraction");

    Matrix result(rows_, cols_);

    size_t n = data.size();

    for (size_t i = 0; i < n; i++)
        result.data[i] = data[i] - other.data[i];

    return result;
}


/* Scalar Multiplication */

Matrix Matrix::operator*(double scalar) const
{
    Matrix result(rows_, cols_);

    size_t n = data.size();

    for (size_t i = 0; i < n; i++)
        result.data[i] = data[i] * scalar;

    return result;
}


/* Matrix-Vector Multiplication */

Vector Matrix::operator*(const Vector& vec) const
{
    if (cols_ != vec.size())
        throw std::invalid_argument("Matrix columns must match vector size");

    Vector result(rows_);

    for (size_t i = 0; i < rows_; i++)
    {
        double sum = 0.0;

        for (size_t j = 0; j < cols_; j++)
            sum += data[i * cols_ + j] * vec[j];

        result[i] = sum;
    }

    return result;
}


/* Matrix-Matrix Multiplication */

Matrix Matrix::operator*(const Matrix& other) const
{
    if (cols_ != other.rows_)
        throw std::invalid_argument("Matrix dimensions invalid for multiplication");

    Matrix result(rows_, other.cols_);

    for (size_t i = 0; i < rows_; i++)
    {
        for (size_t j = 0; j < other.cols_; j++)
        {
            double sum = 0.0;

            for (size_t k = 0; k < cols_; k++)
                sum += data[i * cols_ + k] * other.data[k * other.cols_ + j];

            result.data[i * other.cols_ + j] = sum;
        }
    }

    return result;
}


/* Transpose */

Matrix Matrix::transpose() const
{
    Matrix result(cols_, rows_);

    for (size_t i = 0; i < rows_; i++)
        for (size_t j = 0; j < cols_; j++)
            result.data[j * rows_ + i] = data[i * cols_ + j];

    return result;
}


/* Row Extraction */

Vector Matrix::row(size_t r) const
{
    if (r >= rows_)
        throw std::out_of_range("Row index out of range");

    Vector result(cols_);

    for (size_t j = 0; j < cols_; j++)
        result[j] = data[r * cols_ + j];

    return result;
}


/* Column Extraction */

Vector Matrix::col(size_t c) const
{
    if (c >= cols_)
        throw std::out_of_range("Column index out of range");

    Vector result(rows_);

    for (size_t i = 0; i < rows_; i++)
        result[i] = data[i * cols_ + c];

    return result;
}


/* Print */

void Matrix::print() const
{
    for (size_t i = 0; i < rows_; i++)
    {
        std::cout << "[ ";

        for (size_t j = 0; j < cols_; j++)
            std::cout << data[i * cols_ + j] << " ";

        std::cout << "]\n";
    }
}


/* Identity Matrix */

Matrix Matrix::identity(size_t n)
{
    Matrix I(n, n);

    for (size_t i = 0; i < n; i++)
        I.data[i * n + i] = 1.0;

    return I;
}


/* Sum */

double Matrix::sum() const
{
    double s = 0.0;

    for (double v : data)
        s += v;

    return s;
}


/* Mean */

double Matrix::mean() const
{
    if (data.empty())
        throw std::runtime_error("Cannot compute mean of empty matrix");

    return sum() / data.size();
}


/* Hadamard Product */

Matrix Matrix::hadamard(const Matrix& other) const
{
    if (rows_ != other.rows_ || cols_ != other.cols_)
        throw std::invalid_argument("Matrix dimensions must match for Hadamard product");

    Matrix result(rows_, cols_);

    size_t n = data.size();

    for (size_t i = 0; i < n; i++)
        result.data[i] = data[i] * other.data[i];

    return result;
}


/* Normalize */

Matrix Matrix::normalize() const
{
    if (data.empty())
        throw std::runtime_error("Cannot normalize empty matrix");

    double minVal = data[0];
    double maxVal = data[0];

    for (double v : data)
    {
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
    }

    double range = maxVal - minVal;

    if (range == 0.0)
        return Matrix(rows_, cols_, 0.0);

    Matrix result(rows_, cols_);

    size_t n = data.size();

    for (size_t i = 0; i < n; i++)
        result.data[i] = (data[i] - minVal) / range;

    return result;
}


/* Scalar * Matrix */

Matrix operator*(double scalar, const Matrix& mat)
{
    return mat * scalar;
}


/* Determinant */

double Matrix::determinant() const
{
    if (rows_ != cols_)
        throw std::invalid_argument("Determinant requires a square matrix");

    if (rows_ == 1)
        return data[0];

    if (rows_ == 2)
        return data[0] * data[3] - data[1] * data[2];

    double det = 0.0;

    for (size_t col = 0; col < cols_; col++)
    {
        double sign = (col & 1) ? -1.0 : 1.0;
        det += sign * (*this)(0, col) * minorMatrix(0, col).determinant();
    }

    return det;
}


/* Minor Matrix */

Matrix Matrix::minorMatrix(size_t row, size_t col) const
{
    Matrix minor(rows_ - 1, cols_ - 1);

    size_t r = 0;

    for (size_t i = 0; i < rows_; i++)
    {
        if (i == row) continue;

        size_t c = 0;

        for (size_t j = 0; j < cols_; j++)
        {
            if (j == col) continue;

            minor(r, c) = (*this)(i, j);
            c++;
        }

        r++;
    }

    return minor;
}


/* Matrix Inverse */

Matrix Matrix::inverse() const
{
    if (rows_ != cols_)
        throw std::invalid_argument("Inverse requires a square matrix");

    double det = determinant();

    if (det == 0)
        throw std::runtime_error("Matrix is singular and cannot be inverted");

    Matrix cofactor(rows_, cols_);

    for (size_t i = 0; i < rows_; i++)
        for (size_t j = 0; j < cols_; j++)
        {
            double sign = ((i + j) & 1) ? -1.0 : 1.0;
            cofactor(i, j) = sign * minorMatrix(i, j).determinant();
        }

    Matrix adjugate = cofactor.transpose();

    return adjugate * (1.0 / det);
}


/* Trace */

double Matrix::trace() const
{
    if (rows_ != cols_)
        throw std::invalid_argument("Trace requires square matrix");

    double t = 0.0;

    for (size_t i = 0; i < rows_; i++)
        t += data[i * cols_ + i];

    return t;
}


/* Column Mean */

Vector Matrix::columnMean() const
{
    if (rows_ == 0)
        throw std::runtime_error("Cannot compute column mean of empty matrix");

    Vector mean(cols_);

    for (size_t j = 0; j < cols_; j++)
    {
        double sum = 0.0;

        for (size_t i = 0; i < rows_; i++)
            sum += data[i * cols_ + j];

        mean[j] = sum / rows_;
    }

    return mean;
}


/* Standardize */

Matrix Matrix::standardize() const
{
    Vector mean = columnMean();
    Vector std(cols_);

    for (size_t j = 0; j < cols_; j++)
    {
        double variance = 0.0;

        for (size_t i = 0; i < rows_; i++)
        {
            double diff = data[i * cols_ + j] - mean[j];
            variance += diff * diff;
        }

        std[j] = std::sqrt(variance / rows_);
    }

    Matrix result(rows_, cols_);

    for (size_t i = 0; i < rows_; i++)
        for (size_t j = 0; j < cols_; j++)
        {
            if (std[j] == 0)
                result(i, j) = 0;
            else
                result(i, j) = (data[i * cols_ + j] - mean[j]) / std[j];
        }

    return result;
}


/* Covariance */

Matrix Matrix::covariance() const
{
    if (rows_ < 2)
        throw std::runtime_error("Need at least two samples for covariance");

    Matrix centered = *this;
    Vector mean = columnMean();

    for (size_t i = 0; i < rows_; i++)
        for (size_t j = 0; j < cols_; j++)
            centered(i, j) -= mean[j];

    Matrix Xt = centered.transpose();

    return (Xt * centered) * (1.0 / (rows_ - 1));
}


/* Correlation */

Matrix Matrix::correlation() const
{
    Matrix cov = covariance();

    Matrix corr(cols_, cols_);
    Vector std(cols_);

    for (size_t j = 0; j < cols_; j++)
        std[j] = std::sqrt(cov(j, j));

    for (size_t i = 0; i < cols_; i++)
        for (size_t j = 0; j < cols_; j++)
        {
            if (std[i] == 0 || std[j] == 0)
                corr(i, j) = 0;
            else
                corr(i, j) = cov(i, j) / (std[i] * std[j]);
        }

    return corr;
}