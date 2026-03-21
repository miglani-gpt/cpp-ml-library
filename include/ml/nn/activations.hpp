#pragma once

#include "ml/linalg/vector.hpp"
#include "ml/linalg/matrix.hpp"

namespace ml::nn::activations {

    // -------- Scalar --------
    double relu(double x);
    double sigmoid(double x);
    double tanh(double x);

    // -------- Vector --------
    Vector relu(const Vector& v);
    Vector sigmoid(const Vector& v);
    Vector tanh(const Vector& v);

    // -------- Matrix --------
    Matrix relu(const Matrix& m);
    Matrix sigmoid(const Matrix& m);
    Matrix tanh(const Matrix& m);

    // -------- Derivatives (Scalar) --------
    double relu_derivative(double x);
    double sigmoid_derivative(double x);
    double tanh_derivative(double x);

}