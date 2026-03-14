cmake_minimum_required(VERSION 3.10)

project(cpp_ml_library)

# Use modern C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =====================================
# Library
# =====================================

add_library(ml
    src/linalg/vector.cpp
    src/linalg/matrix.cpp
)

# Expose include directory
target_include_directories(ml PUBLIC include)

# =====================================
# Examples
# =====================================

add_executable(vector_example
    examples/vector_example.cpp
)

target_link_libraries(vector_example ml)

add_executable(matrix_example
    examples/matrix_example.cpp
)

target_link_libraries(matrix_example ml)
