#include "matmul_kernel.h"

#include <cstddef>
#include <cstdint>

template <typename T>
void matmul_kernel(const T* a, const T* b, T* c, std::size_t m, std::size_t k, std::size_t n) {
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            T acc = static_cast<T>(0);
            for (std::size_t kk = 0; kk < k; ++kk) {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

// Explicit instantiations for the dtypes used in this example.
template void matmul_kernel<float>(const float*, const float*, float*, std::size_t, std::size_t, std::size_t);
template void matmul_kernel<int32_t>(const int32_t*, const int32_t*, int32_t*, std::size_t, std::size_t, std::size_t);
