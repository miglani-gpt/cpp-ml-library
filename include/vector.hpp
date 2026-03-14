#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <vector> //standard vector library not the vector class we implement


class Vector {
private:
    std::vector<double> data;

public:
    Vector(int size);

    int size() const;

    double get(int index) const;
    void set(int index, double value);

    double dot(const Vector& other) const;
};

#endif
