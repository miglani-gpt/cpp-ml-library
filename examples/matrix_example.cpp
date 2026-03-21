#include <iostream>
#include "ml/linalg/vector.hpp"
#include "ml/linalg/matrix.hpp"

int main() {
    std::cout << "===== MATRIX EXAMPLE =====\n\n";

    Matrix A(3, 3);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;
    A(2,0) = 7; A(2,1) = 8; A(2,2) = 10;

    Matrix B(3, 3);
    B(0,0) = 1; B(0,1) = 0; B(0,2) = 2;
    B(1,0) = -1; B(1,1) = 3; B(1,2) = 1;
    B(2,0) = 4; B(2,1) = 1; B(2,2) = 0;

    std::cout << "Matrix A:\n";
    A.print();

    std::cout << "\nMatrix B:\n";
    B.print();

    std::cout << "\nBasic operations:\n";
    std::cout << "A + B:\n";
    (A + B).print();

    std::cout << "\nA - B:\n";
    (A - B).print();

    std::cout << "\nA * 2:\n";
    (A * 2.0).print();

    std::cout << "\n2 * A:\n";
    (2.0 * A).print();

    std::cout << "\nA / 2:\n";
    (A / 2.0).print();

    std::cout << "\nIn-place operations:\n";
    Matrix C = A;
    C += B;
    std::cout << "C = A; C += B:\n";
    C.print();

    C = A;
    C -= B;
    std::cout << "\nC = A; C -= B:\n";
    C.print();

    C = A;
    C *= 3.0;
    std::cout << "\nC = A; C *= 3:\n";
    C.print();

    C = A;
    C /= 2.0;
    std::cout << "\nC = A; C /= 2:\n";
    C.print();

    std::cout << "\nMatrix × Vector:\n";
    Vector x = {1.0, 2.0, 3.0};
    std::cout << "x: ";
    x.print();
    std::cout << "A * x:\n";
    (A * x).print();

    std::cout << "\nMatrix × Matrix:\n";
    Matrix AB = A * B;
    AB.print();

    std::cout << "\nTranspose(A):\n";
    A.transpose().print();

    std::cout << "\nShape/stat utilities:\n";
    std::cout << "rows(A): " << A.rows() << '\n';
    std::cout << "cols(A): " << A.cols() << '\n';
    std::cout << "sum(A): " << A.sum() << '\n';
    std::cout << "mean(A): " << A.mean() << '\n';
    std::cout << "trace(A): " << A.trace() << '\n';

    std::cout << "\nRow 1 of A:\n";
    A.row(1).print();

    std::cout << "Col 2 of A:\n";
    A.col(2).print();

    std::cout << "\nColumn mean of A:\n";
    A.columnMean().print();

    std::cout << "\nNormalize(A):\n";
    A.normalize().print();

    std::cout << "\nStandardize(A):\n";
    A.standardize().print();

    std::cout << "\nHadamard(A, B):\n";
    A.hadamard(B).print();

    std::cout << "\nApply (square each element):\n";
    A.apply([](double z) { return z * z; }).print();

    std::cout << "\nAxis-wise sum:\n";
    std::cout << "sum(axis=0): ";
    A.sum(0).print();
    std::cout << "sum(axis=1): ";
    A.sum(1).print();

    std::cout << "\nAxis-wise mean:\n";
    std::cout << "mean(axis=0): ";
    A.mean(0).print();
    std::cout << "mean(axis=1): ";
    A.mean(1).print();

    std::cout << "\nCovariance(A):\n";
    A.covariance().print();

    std::cout << "\nCorrelation(A):\n";
    A.correlation().print();

    std::cout << "\nDeterminant and inverse test:\n";
    Matrix D(2, 2);
    D(0,0) = 4; D(0,1) = 7;
    D(1,0) = 2; D(1,1) = 6;

    std::cout << "D:\n";
    D.print();

    std::cout << "det(D): " << D.determinant() << '\n';
    std::cout << "inv(D):\n";
    D.inverse().print();

    std::cout << "\nIdentity(3):\n";
    Matrix::identity(3).print();

    return 0;
}