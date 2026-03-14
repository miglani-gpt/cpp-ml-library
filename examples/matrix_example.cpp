#include "../include/matrix.hpp"
#include "../include/vector.hpp"

#include <iostream>

int main() {

    std::cout << "=== Matrix Creation ===\n";

    Matrix A(2, 3);

    A.set(0,0,1);
    A.set(0,1,2);
    A.set(0,2,3);

    A.set(1,0,4);
    A.set(1,1,5);
    A.set(1,2,6);

    std::cout << "Matrix A:\n";
    A.print();


    Matrix B(2,3,1.0);

    std::cout << "\nMatrix B (initialized with 1s):\n";
    B.print();


    std::cout << "\n=== Element Access ===\n";

    std::cout << "A(1,2) = " << A(1,2) << std::endl;

    A(1,2) = 10;

    std::cout << "After modification A:\n";
    A.print();


    std::cout << "\n=== Matrix Addition ===\n";

    Matrix C = A + B;

    C.print();


    std::cout << "\n=== Matrix Subtraction ===\n";

    Matrix D = A - B;

    D.print();


    std::cout << "\n=== Scalar Multiplication ===\n";

    Matrix E = A * 2.0;

    E.print();


    std::cout << "\n=== Matrix Vector Multiplication ===\n";

    Vector v(3);
    v.set(0,1);
    v.set(1,2);
    v.set(2,3);

    Vector result = A * v;

    std::cout << "Vector result:\n";
    result.print();


    std::cout << "\n=== Matrix Matrix Multiplication ===\n";

    Matrix M1(2,3);

    M1.set(0,0,1);
    M1.set(0,1,2);
    M1.set(0,2,3);

    M1.set(1,0,4);
    M1.set(1,1,5);
    M1.set(1,2,6);


    Matrix M2(3,2);

    M2.set(0,0,7);
    M2.set(0,1,8);

    M2.set(1,0,9);
    M2.set(1,1,10);

    M2.set(2,0,11);
    M2.set(2,1,12);

    Matrix M3 = M1 * M2;

    std::cout << "Result of M1 * M2:\n";
    M3.print();


    std::cout << "\n=== Transpose ===\n";

    Matrix At = A.transpose();

    At.print();


    std::cout << "\n=== Row Extraction ===\n";

    Vector r = A.row(1);

    r.print();


    std::cout << "\n=== Column Extraction ===\n";

    Vector c = A.col(2);

    c.print();


    std::cout << "\n=== Program Completed Successfully ===\n";

    return 0;
}