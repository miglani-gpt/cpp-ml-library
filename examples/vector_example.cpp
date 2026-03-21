#include "ml/linalg/vector.hpp"
#include <iostream>

int main()
{
    std::cout << "=== Input Feature Vector ===\n";

    // Feature vector (example: 5 features)
    Vector x(5);
    for (size_t i = 0; i < x.size(); i++)
        x[i] = i + 1;   // [1, 2, 3, 4, 5]

    std::cout << "x: ";
    x.print();


    std::cout << "\n=== Weight Initialization ===\n";

    // Weight vector
    Vector w(5, 2.0);

    std::cout << "w: ";
    w.print();


    std::cout << "\n=== Basic Operations ===\n";

    std::cout << "x + w: ";
    (x + w).print();

    std::cout << "x - w: ";
    (x - w).print();

    std::cout << "x * 3: ";
    (x * 3.0).print();

    std::cout << "2 * x: ";
    (2.0 * x).print();


    std::cout << "\n=== Prediction (Linear Model) ===\n";

    double bias = 1.5;

    double y_pred = x.dot(w) + bias;

    std::cout << "Prediction (x·w + b): " << y_pred << "\n";


    std::cout << "\n=== Statistics ===\n";

    std::cout << "sum(x): " << x.sum() << "\n";
    std::cout << "mean(x): " << x.mean() << "\n";
    std::cout << "argmax(x): " << x.argmax() << "\n";
    std::cout << "argmin(x): " << x.argmin() << "\n";


    std::cout << "\n=== Normalization ===\n";

    Vector x_norm = x.normalize();

    std::cout << "normalized x: ";
    x_norm.print();


    std::cout << "\n=== Element-wise Operations (Gradient-style) ===\n";

    Vector grad(5, 0.1);

    std::cout << "gradient: ";
    grad.print();

    Vector updated_w = w - (grad * 0.01);

    std::cout << "updated weights: ";
    updated_w.print();


    return 0;
}