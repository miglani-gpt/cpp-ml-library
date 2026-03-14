#include "ml/linalg/vector.hpp"
#include <iostream>

int main()
{
    std::cout << "=== Vector Construction ===\n";

    Vector v1(5);
    Vector v2(5, 2.0);

    for (size_t i = 0; i < v1.size(); i++)
        v1[i] = i + 1;

    std::cout << "v1: ";
    v1.print();

    std::cout << "v2: ";
    v2.print();


    std::cout << "\n=== Vector Arithmetic ===\n";

    Vector add = v1 + v2;
    Vector sub = v1 - v2;
    Vector mul = v1 * 3.0;
    Vector mul2 = 2.0 * v1;

    std::cout << "v1 + v2: ";
    add.print();

    std::cout << "v1 - v2: ";
    sub.print();

    std::cout << "v1 * 3: ";
    mul.print();

    std::cout << "2 * v1: ";
    mul2.print();


    std::cout << "\n=== Dot Product & Norm ===\n";

    double dot = v1.dot(v2);
    double norm = v1.norm();

    std::cout << "dot(v1, v2): " << dot << "\n";
    std::cout << "||v1||: " << norm << "\n";


    std::cout << "\n=== Statistical Utilities ===\n";

    std::cout << "sum(v1): " << v1.sum() << "\n";
    std::cout << "mean(v1): " << v1.mean() << "\n";
    std::cout << "argmax(v1): " << v1.argmax() << "\n";
    std::cout << "argmin(v1): " << v1.argmin() << "\n";


    std::cout << "\n=== Normalization ===\n";

    Vector norm_v = v1.normalize();

    std::cout << "normalized v1: ";
    norm_v.print();

    return 0;
}