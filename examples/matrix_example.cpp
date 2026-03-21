#include "ml/linalg/matrix.hpp"
#include <iostream>

int main()
{
    std::cout << "=== Dataset (Feature Matrix X) ===\n";

    // 3 samples, 3 features
    Matrix X(3,3);

    X(0,0)=1; X(0,1)=2; X(0,2)=3;
    X(1,0)=4; X(1,1)=5; X(1,2)=6;
    X(2,0)=7; X(2,1)=8; X(2,2)=9;

    std::cout << "X:\n";
    X.print();


    std::cout << "\n=== Weights Initialization ===\n";

    Vector w(3, 1.0);   // 3 features → 3 weights
    double bias = 0.5;

    std::cout << "w: ";
    w.print();


    std::cout << "\n=== Forward Pass (Prediction) ===\n";

    // y_pred = Xw + b
    Vector y_pred = X * w + bias;

    std::cout << "Predictions:\n";
    y_pred.print();


    std::cout << "\n=== Basic Matrix Ops ===\n";

    std::cout << "X + 1:\n";
    (X + 1.0).print();

    std::cout << "X * 2:\n";
    (X * 2.0).print();

    std::cout << "Transpose(X):\n";
    X.transpose().print();


    std::cout << "\n=== Feature-wise Statistics ===\n";

    std::cout << "Column Mean:\n";
    X.mean(0).print();

    std::cout << "Row Sum:\n";
    X.sum(1).print();


    std::cout << "\n=== Normalization / Standardization ===\n";

    Matrix X_norm = X.normalize();
    std::cout << "Normalized X:\n";
    X_norm.print();

    Matrix X_std = X.standardize();
    std::cout << "Standardized X:\n";
    X_std.print();


    std::cout << "\n=== Matrix Multiplication ===\n";

    Matrix Xt = X.transpose();

    Matrix XtX = Xt * X;

    std::cout << "X^T * X:\n";
    XtX.print();


    std::cout << "\n=== Covariance & Correlation ===\n";

    std::cout << "Covariance Matrix:\n";
    X.covariance().print();

    std::cout << "Correlation Matrix:\n";
    X.correlation().print();


    std::cout << "\n=== Small Linear System (Determinant & Inverse) ===\n";

    Matrix D(2,2);
    D(0,0)=4; D(0,1)=7;
    D(1,0)=2; D(1,1)=6;

    std::cout << "D:\n";
    D.print();

    std::cout << "det(D): " << D.determinant() << "\n";

    std::cout << "D^-1:\n";
    D.inverse().print();


    std::cout << "\n=== Gradient-style Update (Preview) ===\n";

    Vector grad(3, 0.1);

    w -= grad * 0.01;

    std::cout << "Updated weights: ";
    w.print();


    return 0;
}