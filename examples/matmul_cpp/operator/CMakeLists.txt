cmake_minimum_required(VERSION 3.10)
project(matmul_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul_runner matmul_runner.cpp matmul_kernel.cpp)
