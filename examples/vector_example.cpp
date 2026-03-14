#include "../include/vector.hpp"
#include <iostream>

int main() {

    std::cout << "=== Vector Creation ===\n";

    Vector v1(3);
    v1.set(0, 1.0);
    v1.set(1, 2.0);
    v1.set(2, 3.0);

    Vector v2(3, 4.0);

    std::cout << "v1 = ";
    v1.print();

    std::cout << "v2 = ";
    v2.print();


    std::cout << "\n=== Element Access ===\n";

    std::cout << "v1[1] = " << v1[1] << std::endl;

    v1[1] = 10.0;

    std::cout << "After modifying v1[1]: ";
    v1.print();


    std::cout << "\n=== Vector Addition ===\n";

    Vector v3 = v1 + v2;

    std::cout << "v1 + v2 = ";
    v3.print();


    std::cout << "\n=== Vector Subtraction ===\n";

    Vector v4 = v1 - v2;

    std::cout << "v1 - v2 = ";
    v4.print();


    std::cout << "\n=== Scalar Multiplication ===\n";

    Vector v5 = v1 * 2.0;

    std::cout << "v1 * 2 = ";
    v5.print();

    Vector v6 = 3.0 * v1;

    std::cout << "3 * v1 = ";
    v6.print();


    std::cout << "\n=== Dot Product ===\n";

    double dotProduct = v1.dot(v2);

    std::cout << "v1 . v2 = " << dotProduct << std::endl;


    std::cout << "\n=== Norm ===\n";

    std::cout << "||v1|| = " << v1.norm() << std::endl;


    std::cout << "\n=== Sum and Mean ===\n";

    std::cout << "sum(v1) = " << v1.sum() << std::endl;
    std::cout << "mean(v1) = " << v1.mean() << std::endl;


    std::cout << "\n=== Argmax and Argmin ===\n";

    std::cout << "argmax(v1) = " << v1.argmax() << std::endl;
    std::cout << "argmin(v1) = " << v1.argmin() << std::endl;


    std::cout << "\n=== Normalization ===\n";

    Vector normalized = v1.normalize();

    std::cout << "normalized v1 = ";
    normalized.print();


    std::cout << "\n=== Program Completed Successfully ===\n";

    return 0;
}