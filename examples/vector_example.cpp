#include <iostream>
#include "../include/vector.hpp"

int main() {

    Vector a(3);
    Vector b(3);

    a.set(0,1);
    a.set(1,2);
    a.set(2,3);

    b.set(0,4);
    b.set(1,5);
    b.set(2,6);

    std::cout << "Dot Product: " << a.dot(b) << std::endl;

}
