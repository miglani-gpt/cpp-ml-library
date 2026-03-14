#include "ml/linalg/vector.hpp"

#include <stdexcept>
#include <cmath>
#include <iostream>

/* Constructors */

Vector::Vector(size_t size) : data(size, 0.0) {}

Vector::Vector(size_t size, double initialValue) : data(size, initialValue) {}


/* Size */

size_t Vector::size() const {
    return data.size();
}


/* Element Access */

double Vector::get(size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("Vector index out of range");
    }
    return data[index];
}

void Vector::set(size_t index, double value) {
    if (index >= size()) {
        throw std::out_of_range("Vector index out of range");
    }
    data[index] = value;
}

double& Vector::operator[](size_t index) {
    if (index >= size()) {
        throw std::out_of_range("Vector index out of range");
    }
    return data[index];
}

const double& Vector::operator[](size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("Vector index out of range");
    }
    return data[index];
}


/* Dot Product */

double Vector::dot(const Vector& other) const {

    size_t n = size();

    if (n != other.size()) {
        throw std::invalid_argument("Vector sizes must match for dot product");
    }

    const auto& a = data;
    const auto& b = other.data;

    double result = 0.0;

    for (size_t i = 0; i < n; i++) {
        result += a[i] * b[i];
    }

    return result;
}


/* Vector Norm */

double Vector::norm() const {
    return std::sqrt(dot(*this));
}


/* Vector Addition */

Vector Vector::operator+(const Vector& other) const {

    size_t n = size();

    if (n != other.size()) {
        throw std::invalid_argument("Vector sizes must match for addition");
    }

    Vector result(n);

    for (size_t i = 0; i < n; i++) {
        result[i] = data[i] + other.data[i];
    }

    return result;
}


/* Vector Subtraction */

Vector Vector::operator-(const Vector& other) const {

    size_t n = size();

    if (n != other.size()) {
        throw std::invalid_argument("Vector sizes must match for subtraction");
    }

    Vector result(n);

    for (size_t i = 0; i < n; i++) {
        result[i] = data[i] - other.data[i];
    }

    return result;
}


/* Scalar Multiplication */

Vector Vector::operator*(double scalar) const {

    size_t n = size();

    Vector result(n);

    for (size_t i = 0; i < n; i++) {
        result[i] = data[i] * scalar;
    }

    return result;
}


/* Scalar * Vector */

Vector operator*(double scalar, const Vector& v) {

    size_t n = v.size();

    Vector result(n);

    for (size_t i = 0; i < n; i++) {
        result[i] = scalar * v[i];
    }

    return result;
}


/* Sum */

double Vector::sum() const {

    double total = 0.0;

    for (double v : data) {
        total += v;
    }

    return total;
}


/* Mean */

double Vector::mean() const {

    size_t n = size();

    if (n == 0) {
        throw std::runtime_error("Cannot compute mean of empty vector");
    }

    return sum() / n;
}


/* Argmax */

size_t Vector::argmax() const {

    size_t n = size();

    if (n == 0) {
        throw std::runtime_error("Cannot compute argmax of empty vector");
    }

    size_t index = 0;

    for (size_t i = 1; i < n; i++) {
        if (data[i] > data[index]) {
            index = i;
        }
    }

    return index;
}


/* Argmin */

size_t Vector::argmin() const {

    size_t n = size();

    if (n == 0) {
        throw std::runtime_error("Cannot compute argmin of empty vector");
    }

    size_t index = 0;

    for (size_t i = 1; i < n; i++) {
        if (data[i] < data[index]) {
            index = i;
        }
    }

    return index;
}


/* Normalize */

Vector Vector::normalize() const {

    double nrm = norm();

    if (nrm == 0) {
        throw std::runtime_error("Cannot normalize zero vector");
    }

    size_t n = size();

    Vector result(n);

    for (size_t i = 0; i < n; i++) {
        result[i] = data[i] / nrm;
    }

    return result;
}


/* Print */

void Vector::print() const {

    std::cout << "[ ";

    for (double v : data) {
        std::cout << v << " ";
    }

    std::cout << "]\n";
}