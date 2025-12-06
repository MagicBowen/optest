cmake_minimum_required(VERSION 3.10)
project(cann_add_custom CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(add_custom op_runner.cpp)
