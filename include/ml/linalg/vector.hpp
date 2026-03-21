#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <functional>

class Matrix; // forward declaration

class Vector {
private:
    std::vector<double> data;

public:
    /* Constructors */
    explicit Vector(size_t size);
    Vector(size_t size, double initialValue);

    /* Move Semantics */
    Vector(Vector&& other) noexcept;
    Vector& operator=(Vector&& other) noexcept;

    /* Size */
    [[nodiscard]] size_t size() const;

    /* Element Access */
    double get(size_t index) const;
    void set(size_t index, double value);

    double& operator[](size_t index);
    const double& operator[](size_t index) const;

    /* Vector Operations */
    double dot(const Vector& other) const;
    double norm() const;

    Vector operator+(const Vector& other) const;
    Vector operator-(const Vector& other) const;
    Vector operator*(double scalar) const;

    /* In-place operations */
    Vector& operator+=(const Vector& other);
    Vector& operator-=(const Vector& other);
    Vector& operator*=(double scalar);
    Vector& operator/=(double scalar);

    /* Scalar * Vector support */
    friend Vector operator*(double scalar, const Vector& v);

    /* Statistical / ML Utilities */
    double sum() const;
    double mean() const;

    size_t argmax() const;
    size_t argmin() const;

    Vector normalize() const;

    /* Utility */
    void print() const;

    /* ML Ops */
    Vector apply(std::function<double(double)> func) const;
    Vector hadamard(const Vector& other) const;
    Matrix outer(const Vector& other) const;

    /* Scalar ops */
    Vector operator+(double scalar) const;
    Vector operator-(double scalar) const;
    Vector operator/(double scalar) const;
};

#endif