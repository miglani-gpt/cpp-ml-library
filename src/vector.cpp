#include "../include/vector.hpp"
#include <stdexcept>

Vector::Vector(int size) {
    data.resize(size);
}

int Vector::size() const {
    return data.size();
}

double Vector::get(int index) const {
    return data[index];
}

void Vector::set(int index, double value) {
    data[index] = value;
}

double Vector::dot(const Vector& other) const {

    if (data.size() != other.size()) {
        throw std::invalid_argument("Vector sizes must match for dot product");
    }

    double result = 0;

    for (size_t i = 0; i < data.size(); i++) {
        result += data[i] * other.data[i];
    }

    return result;
}
