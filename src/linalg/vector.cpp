#include "ml/linalg/vector.hpp"
#include "ml/linalg/matrix.hpp"

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <algorithm>

// ================================
// Constructors
// ================================

Vector::Vector(size_t size)
    : data(size, 0.0) {}

Vector::Vector(size_t size, double initialValue)
    : data(size, initialValue) {}

Vector::Vector(std::initializer_list<double> list)
    : data(list) {}

// ================================
// Move Semantics
// ================================

Vector::Vector(Vector&& other) noexcept
    : data(std::move(other.data)) {}

Vector& Vector::operator=(Vector&& other) noexcept {
    if (this != &other) {
        data = std::move(other.data);
    }
    return *this;
}

// ================================
// Copy Semantics
// ================================

Vector::Vector(const Vector& other)
    : data(other.data) {}

Vector& Vector::operator=(const Vector& other) {
    if (this != &other) {
        data = other.data;
    }
    return *this;
}

// ================================
// Size
// ================================

size_t Vector::size() const {
    return data.size();
}

// ================================
// Element Access
// ================================

double Vector::get(size_t index) const {
    if (index >= data.size()) {
        throw std::out_of_range("Vector index out of range");
    }
    return data[index];
}

void Vector::set(size_t index, double value) {
    if (index >= data.size()) {
        throw std::out_of_range("Vector index out of range");
    }
    data[index] = value;
}

double& Vector::operator[](size_t index) {
    return data[index];
}

const double& Vector::operator[](size_t index) const {
    return data[index];
}

// ================================
// Vector Operations
// ================================

double Vector::dot(const Vector& other) const {
    if (size() != other.size()) {
        throw std::invalid_argument("Dot product size mismatch");
    }

    double result = 0.0;
    for (size_t i = 0; i < size(); ++i) {
        result += data[i] * other.data[i];
    }
    return result;
}

double Vector::norm() const {
    return std::sqrt(dot(*this));
}

Vector Vector::operator+(const Vector& other) const {
    if (size() != other.size()) {
        throw std::invalid_argument("Vector addition size mismatch");
    }

    Vector result(size());
    for (size_t i = 0; i < size(); ++i) {
        result[i] = data[i] + other.data[i];
    }
    return result;
}

Vector Vector::operator-(const Vector& other) const {
    if (size() != other.size()) {
        throw std::invalid_argument("Vector subtraction size mismatch");
    }

    Vector result(size());
    for (size_t i = 0; i < size(); ++i) {
        result[i] = data[i] - other.data[i];
    }
    return result;
}

Vector Vector::operator*(double scalar) const {
    Vector result(size());
    for (size_t i = 0; i < size(); ++i) {
        result[i] = data[i] * scalar;
    }
    return result;
}

// ================================
// In-place operations
// ================================

Vector& Vector::operator+=(const Vector& other) {
    if (size() != other.size()) {
        throw std::invalid_argument("Vector += size mismatch");
    }

    for (size_t i = 0; i < size(); ++i) {
        data[i] += other.data[i];
    }
    return *this;
}

Vector& Vector::operator-=(const Vector& other) {
    if (size() != other.size()) {
        throw std::invalid_argument("Vector -= size mismatch");
    }

    for (size_t i = 0; i < size(); ++i) {
        data[i] -= other.data[i];
    }
    return *this;
}

Vector& Vector::operator*=(double scalar) {
    for (double& v : data) {
        v *= scalar;
    }
    return *this;
}

Vector& Vector::operator/=(double scalar) {
    if (scalar == 0.0) {
        throw std::invalid_argument("Division by zero");
    }

    for (double& v : data) {
        v /= scalar;
    }
    return *this;
}

// ================================
// Scalar * Vector
// ================================

Vector operator*(double scalar, const Vector& v) {
    return v * scalar;
}

// ================================
// Statistical / ML Utilities
// ================================

double Vector::sum() const {
    double s = 0.0;
    for (double v : data) {
        s += v;
    }
    return s;
}

double Vector::mean() const {
    if (data.empty()) {
        throw std::invalid_argument("Mean of empty vector is undefined");
    }
    return sum() / data.size();
}

size_t Vector::argmax() const {
    if (data.empty()) {
        throw std::invalid_argument("argmax of empty vector");
    }

    size_t idx = 0;
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i] > data[idx]) {
            idx = i;
        }
    }
    return idx;
}

size_t Vector::argmin() const {
    if (data.empty()) {
        throw std::invalid_argument("argmin of empty vector");
    }

    size_t idx = 0;
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i] < data[idx]) {
            idx = i;
        }
    }
    return idx;
}

Vector Vector::normalize() const {
    if (data.empty()) {
        return Vector(0);
    }

    double min_val = data[0];
    double max_val = data[0];

    for (double v : data) {
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }

    double range = max_val - min_val;
    Vector result(size(), 0.0);

    if (range == 0.0) {
        return result;
    }

    for (size_t i = 0; i < size(); ++i) {
        result[i] = (data[i] - min_val) / range;
    }

    return result;
}

// ================================
// Utility
// ================================

void Vector::print() const {
    for (size_t i = 0; i < size(); ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << '\n';
}

// ================================
// ML Ops
// ================================

Vector Vector::hadamard(const Vector& other) const {
    if (size() != other.size()) {
        throw std::invalid_argument("Hadamard product size mismatch");
    }

    Vector result(size());
    for (size_t i = 0; i < size(); ++i) {
        result[i] = data[i] * other.data[i];
    }
    return result;
}

Matrix Vector::outer(const Vector& other) const {
    Matrix result(size(), other.size(), 0.0);

    for (size_t i = 0; i < size(); ++i) {
        for (size_t j = 0; j < other.size(); ++j) {
            result(i, j) = data[i] * other.data[j];
        }
    }

    return result;
}

// ================================
// Scalar Ops
// ================================

Vector Vector::operator+(double scalar) const {
    Vector result(size());
    for (size_t i = 0; i < size(); ++i) {
        result[i] = data[i] + scalar;
    }
    return result;
}

Vector Vector::operator-(double scalar) const {
    Vector result(size());
    for (size_t i = 0; i < size(); ++i) {
        result[i] = data[i] - scalar;
    }
    return result;
}

Vector Vector::operator/(double scalar) const {
    if (scalar == 0.0) {
        throw std::invalid_argument("Division by zero");
    }

    Vector result(size());
    for (size_t i = 0; i < size(); ++i) {
        result[i] = data[i] / scalar;
    }
    return result;
}