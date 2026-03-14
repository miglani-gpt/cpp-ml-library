#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <vector>
#include <cstddef>
#include <stdexcept>

class Vector {
private:
    std::vector<double> data; // internal storage for vector elements

public:
    /* Constructors */
    explicit Vector(size_t size);
    Vector(size_t size, double initialValue);

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
};

#endif