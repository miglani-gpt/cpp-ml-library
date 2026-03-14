# C++ Machine Learning Library

![C++](https://img.shields.io/badge/language-C++-blue)
![Build](https://img.shields.io/badge/build-CMake-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-in%20development-orange)

A lightweight **machine learning library built from scratch in C++** designed to understand how real ML frameworks work internally.

The project focuses on implementing:

- Linear algebra from scratch
- Machine learning algorithms
- Optimization techniques
- Statistical utilities

All without relying on external ML frameworks.

The goal is to understand **how real machine learning libraries are implemented internally**.

---

# Demo

*(Optional — add a GIF later)*

Example placeholder:

![demo](docs/demo.gif)

You can record a short terminal demo and place it in `docs/demo.gif`.

---

# Table of Contents

- Overview
- Features
- Architecture
- Implemented Components
- Project Structure
- Example Usage
- Build Instructions
- Roadmap
- Learning Objectives
- License

---

# Overview

Most machine learning libraries hide the mathematical implementation details behind high-level APIs.

This project aims to:

- implement ML algorithms **from scratch**
- build a **custom linear algebra engine**
- understand **numerical computation**
- design **modular ML systems**
- explore how **real ML libraries are structured**

The library acts as a **mini machine learning framework inspired by Scikit-Learn**, implemented entirely in **modern C++**.

---

# Features

## Custom Linear Algebra Engine

All machine learning computations are built on a custom **Vector and Matrix library**.

### Matrix Features

- Matrix creation
- Matrix addition and subtraction
- Scalar multiplication
- Matrix multiplication
- Matrix–vector multiplication
- Matrix transpose
- Row and column extraction
- Determinant computation
- Matrix inverse
- Hadamard product
- Matrix normalization

### Vector Features

- Dot product
- Vector norm
- Vector normalization
- Sum and mean
- Argmax / argmin

---

## Statistical Utilities

Common utilities used in machine learning pipelines:

- Trace
- Column mean
- Feature standardization
- Covariance matrix
- Correlation matrix

---

# Architecture

The library follows a layered architecture similar to real machine learning frameworks.

```
        Linear Algebra Engine
                │
                ▼
        Statistical Utilities
                │
                ▼
        Machine Learning Algorithms
                │
                ▼
        Evaluation / Data Utilities
```

Core design principles:

- minimal dependencies
- modular architecture
- educational implementation
- efficient numerical operations

---

# Implemented Components

## Linear Algebra

Core numerical engine:

- Vector class
- Matrix class
- Matrix operations
- Statistical matrix utilities

These components act as the **foundation for all ML algorithms**.

---

## Machine Learning Algorithms (Planned / In Progress)

### Linear Regression

Predicts continuous values using a linear relationship between variables.

Model:

```
y = wX + b
```

Functions:

```
fit(X, y)
predict(X)
```

---

### Logistic Regression

Binary classification using sigmoid activation.

Functions:

```
fit(X, y)
predict(X)
predict_proba(X)
```

---

### K-Means Clustering

Unsupervised clustering algorithm.

Algorithm steps:

1. Initialize cluster centroids
2. Assign points to nearest centroid
3. Update centroids
4. Repeat until convergence

---

### Principal Component Analysis (PCA)

Dimensionality reduction technique.

Steps:

1. Standardize data
2. Compute covariance matrix
3. Extract principal components
4. Project data onto new feature space

---

# Project Structure

```
cpp-ml-library
│
├── include/                     # Public library headers
│   └── ml/
│       ├── linalg/              # Linear algebra engine
│       │   ├── vector.hpp
│       │   └── matrix.hpp
│       │
│       ├── models/              # ML algorithms
│       │
│       ├── stats/               # Statistical utilities
│       │
│       └── optim/               # Optimization algorithms
│
├── src/                         # Library implementations
│   ├── linalg/
│   │   ├── vector.cpp
│   │   └── matrix.cpp
│   │
│   ├── models/
│   ├── stats/
│   └── optim/
│
├── examples/                    # Example programs
│   ├── vector_example.cpp
│   └── matrix_example.cpp
│
├── tests/                       # Unit tests
│
├── data/                        # Sample datasets
│
├── docs/                        # Documentation
│
├── build/                       # Build directory (ignored by git)
│
├── CMakeLists.txt               # CMake build configuration
├── README.md
└── .gitignore
```

---

# Example Usage

Example using the matrix library:

```cpp
#include "ml/linalg/vector.hpp"
#include "ml/linalg/matrix.hpp"

int main() {

    Matrix A(2,2);

    A(0,0) = 1;
    A(0,1) = 2;
    A(1,0) = 3;
    A(1,1) = 4;

    Matrix B = A.transpose();

    B.print();

    return 0;
}
```

---

# Build Instructions

This project uses **CMake**.

## Clone repository

```
git clone https://github.com/your-username/cpp-ml-library.git
cd cpp-ml-library
```

## Create build directory

```
mkdir build
cd build
```

## Configure project

```
cmake ..
```

## Compile

```
make
```

## Run examples

```
./vector_example
./matrix_example
```

---

# Roadmap

Planned improvements:

- Linear Regression implementation
- Logistic Regression
- K-Means clustering
- PCA implementation
- Decision Trees
- K-Nearest Neighbors
- Neural Networks
- Model serialization
- Visualization utilities
- GPU acceleration

---

# Learning Objectives

This project demonstrates:

- Linear algebra for machine learning
- Implementation of ML algorithms from scratch
- Numerical optimization techniques
- Modular C++ architecture
- Efficient data structures for ML systems

---

# License

This project is open source and available under the **MIT License**.
