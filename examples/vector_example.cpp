#include <iostream>
#include "ml/linalg/vector.hpp"
#include "ml/linalg/matrix.hpp"

int main() {
    std::cout << "===== VECTOR EXAMPLE =====\n\n";

    Vector v = {1.0, 2.0, 3.0, 4.0};
    Vector w = {4.0, 3.0, 2.0, 1.0};

    std::cout << "v: ";
    v.print();

    std::cout << "w: ";
    w.print();

    std::cout << "\nBasic operations:\n";
    std::cout << "v + w: ";
    (v + w).print();

    std::cout << "v - w: ";
    (v - w).print();

    std::cout << "v * 2: ";
    (v * 2.0).print();

    std::cout << "2 * v: ";
    (2.0 * v).print();

    std::cout << "v / 2: ";
    (v / 2.0).print();

    std::cout << "\nIn-place operations:\n";
    Vector a = v;
    a += w;
    std::cout << "a = v; a += w: ";
    a.print();

    a = v;
    a -= w;
    std::cout << "a = v; a -= w: ";
    a.print();

    a = v;
    a *= 3.0;
    std::cout << "a = v; a *= 3: ";
    a.print();

    a = v;
    a /= 2.0;
    std::cout << "a = v; a /= 2: ";
    a.print();

    std::cout << "\nML/stat utilities:\n";
    std::cout << "sum(v): " << v.sum() << '\n';
    std::cout << "mean(v): " << v.mean() << '\n';
    std::cout << "norm(v): " << v.norm() << '\n';
    std::cout << "argmax(v): " << v.argmax() << '\n';
    std::cout << "argmin(v): " << v.argmin() << '\n';

    std::cout << "normalize(v): ";
    v.normalize().print();

    std::cout << "\nHadamard product:\n";
    std::cout << "v hadamard w: ";
    v.hadamard(w).print();

    std::cout << "\nApply:\n";
    std::cout << "v.apply(x -> x*x): ";
    v.apply([](double x) { return x * x; }).print();

    std::cout << "\nDot product:\n";
    std::cout << "v dot w: " << v.dot(w) << '\n';

    std::cout << "\nOuter product (v outer w):\n";
    Matrix outer = v.outer(w);
    outer.print();

    std::cout << "\nElement access:\n";
    std::cout << "v[2]: " << v[2] << '\n';
    std::cout << "v.get(1): " << v.get(1) << '\n';

    Vector t = v;
    t.set(0, 99.0);
    std::cout << "after t.set(0, 99): ";
    t.print();

    return 0;
}