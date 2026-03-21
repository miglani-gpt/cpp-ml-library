#include <iostream>
#include "ml/nn/activations.hpp"
#include "ml/linalg/vector.hpp"
#include "ml/linalg/matrix.hpp"

int main() {
    std::cout << "===== ACTIVATION EXAMPLE =====\n\n";

    std::cout << "Scalar tests:\n";
    std::cout << "relu(-5) = " << ml::nn::activations::relu(-5.0) << '\n';
    std::cout << "relu(3) = " << ml::nn::activations::relu(3.0) << '\n';
    std::cout << "sigmoid(0) = " << ml::nn::activations::sigmoid(0.0) << '\n';
    std::cout << "sigmoid(2) = " << ml::nn::activations::sigmoid(2.0) << '\n';
    std::cout << "tanh(0) = " << ml::nn::activations::tanh(0.0) << '\n';
    std::cout << "tanh(2) = " << ml::nn::activations::tanh(2.0) << '\n';

    std::cout << "\nDerivative tests:\n";
    std::cout << "relu'(3) = " << ml::nn::activations::relu_derivative(3.0) << '\n';
    std::cout << "relu'(-3) = " << ml::nn::activations::relu_derivative(-3.0) << '\n';
    std::cout << "sigmoid'(0) = " << ml::nn::activations::sigmoid_derivative(0.0) << '\n';
    std::cout << "tanh'(0) = " << ml::nn::activations::tanh_derivative(0.0) << '\n';

    Vector v = {-2.0, -1.0, 0.0, 1.0, 2.0};
    std::cout << "\nInput Vector:\n";
    v.print();

    std::cout << "\nVector ReLU:\n";
    ml::nn::activations::relu(v).print();

    std::cout << "Vector Sigmoid:\n";
    ml::nn::activations::sigmoid(v).print();

    std::cout << "Vector Tanh:\n";
    ml::nn::activations::tanh(v).print();

    Matrix M(2, 3);
    M(0,0) = -2.0; M(0,1) = -1.0; M(0,2) = 0.0;
    M(1,0) = 1.0;  M(1,1) = 2.0;  M(1,2) = 3.0;

    std::cout << "\nInput Matrix:\n";
    M.print();

    std::cout << "\nMatrix ReLU:\n";
    ml::nn::activations::relu(M).print();

    std::cout << "Matrix Sigmoid:\n";
    ml::nn::activations::sigmoid(M).print();

    std::cout << "Matrix Tanh:\n";
    ml::nn::activations::tanh(M).print();

    return 0;
}