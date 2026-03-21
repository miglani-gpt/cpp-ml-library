#include "ml/nn/activations.hpp"
#include <cmath>
#include <algorithm>

namespace ml::nn::activations {

    // -------- Scalar --------
    double relu(double x) {
        return std::max(0.0, x);
    }

    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double tanh(double x) {
        return std::tanh(x);
    }

    // -------- Vector --------
    Vector relu(const Vector& v) {
        return v.apply([](double x) { return relu(x); });
    }

    Vector sigmoid(const Vector& v) {
        return v.apply([](double x) { return sigmoid(x); });
    }

    Vector tanh(const Vector& v) {
        return v.apply([](double x) { return tanh(x); });
    }

    // -------- Matrix --------
    Matrix relu(const Matrix& m) {
        return m.apply([](double x) { return relu(x); });
    }

    Matrix sigmoid(const Matrix& m) {
        return m.apply([](double x) { return sigmoid(x); });
    }

    Matrix tanh(const Matrix& m) {
        return m.apply([](double x) { return tanh(x); });
    }

    // -------- Derivatives --------
    double relu_derivative(double x) {
        return x > 0.0 ? 1.0 : 0.0;
    }

    double sigmoid_derivative(double x) {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }

    double tanh_derivative(double x) {
        double t = std::tanh(x);
        return 1.0 - t * t;
    }

}