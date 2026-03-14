#include "ml/linalg/matrix.hpp"
#include <iostream>

int main()
{
    std::cout << "=== Matrix Construction ===\n";

    Matrix A(3,3);

    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;
    A(2,0)=7; A(2,1)=8; A(2,2)=9;

    std::cout << "Matrix A:\n";
    A.print();


    std::cout << "\n=== Identity Matrix ===\n";

    Matrix I = Matrix::identity(3);
    I.print();


    std::cout << "\n=== Matrix Arithmetic ===\n";

    Matrix B(3,3,1.0);

    Matrix add = A + B;
    Matrix sub = A - B;
    Matrix scalar = A * 2.0;
    Matrix scalar2 = 2.0 * A;

    std::cout << "A + B:\n";
    add.print();

    std::cout << "A - B:\n";
    sub.print();

    std::cout << "A * 2:\n";
    scalar.print();

    std::cout << "2 * A:\n";
    scalar2.print();


    std::cout << "\n=== Hadamard Product ===\n";

    Matrix had = A.hadamard(B);
    had.print();


    std::cout << "\n=== Matrix Transpose ===\n";

    Matrix At = A.transpose();
    At.print();


    std::cout << "\n=== Row / Column Extraction ===\n";

    Vector r = A.row(1);
    Vector c = A.col(2);

    std::cout << "Row 1: ";
    r.print();

    std::cout << "Column 2: ";
    c.print();


    std::cout << "\n=== Matrix Vector Multiplication ===\n";

    Vector v(3);

    v[0]=1;
    v[1]=2;
    v[2]=3;

    Vector result = A * v;

    std::cout << "A * v = ";
    result.print();


    std::cout << "\n=== Matrix Multiplication ===\n";

    Matrix C = A * I;

    C.print();


    std::cout << "\n=== Matrix Statistics ===\n";

    std::cout << "Sum: " << A.sum() << "\n";
    std::cout << "Mean: " << A.mean() << "\n";
    std::cout << "Trace: " << A.trace() << "\n";


    std::cout << "\n=== Normalization ===\n";

    Matrix norm = A.normalize();
    norm.print();


    std::cout << "\n=== Determinant ===\n";

    Matrix D(2,2);

    D(0,0)=4; D(0,1)=7;
    D(1,0)=2; D(1,1)=6;

    std::cout << "Matrix D:\n";
    D.print();

    std::cout << "det(D): " << D.determinant() << "\n";


    std::cout << "\n=== Inverse ===\n";

    Matrix inv = D.inverse();

    std::cout << "D^-1:\n";
    inv.print();


    std::cout << "\n=== Standardization ===\n";

    Matrix std = A.standardize();
    std.print();


    std::cout << "\n=== Covariance Matrix ===\n";

    Matrix cov = A.covariance();
    cov.print();


    std::cout << "\n=== Correlation Matrix ===\n";

    Matrix corr = A.correlation();
    corr.print();

    return 0;
}