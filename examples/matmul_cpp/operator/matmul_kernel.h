#pragma once

#include <cstddef>

// Simple CPU matmul: C[m x n] = A[m x k] x B[k x n]
template <typename T>
void matmul_kernel(const T* a, const T* b, T* c, std::size_t m, std::size_t k, std::size_t n);
