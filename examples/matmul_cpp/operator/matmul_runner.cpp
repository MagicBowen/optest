#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "matmul_kernel.h"

namespace {

struct Options {
    std::string dtype = "float32";
    std::string input0 = "data/input0.bin";
    std::string input1 = "data/input1.bin";
    std::string output0 = "out/output0.bin";
    std::string shapes_json;
};

Options parse_args(int argc, char** argv) {
    Options opt{};
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--dtype" || arg == "-t") && i + 1 < argc) {
            opt.dtype = argv[++i];
        } else if (arg == "--input0" && i + 1 < argc) {
            opt.input0 = argv[++i];
        } else if (arg == "--input1" && i + 1 < argc) {
            opt.input1 = argv[++i];
        } else if (arg == "--output0" && i + 1 < argc) {
            opt.output0 = argv[++i];
        } else if (arg == "--shapes" && i + 1 < argc) {
            opt.shapes_json = argv[++i];
        }
    }
    return opt;
}

std::vector<int64_t> extract_numbers(const std::string& text) {
    std::vector<int64_t> values;
    int64_t current = 0;
    bool in_number = false;
    for (char c : text) {
        if (c >= '0' && c <= '9') {
            current = current * 10 + static_cast<int64_t>(c - '0');
            in_number = true;
        } else {
            if (in_number) {
                values.push_back(current);
                current = 0;
                in_number = false;
            }
        }
    }
    if (in_number) {
        values.push_back(current);
    }
    return values;
}

struct MatmulShape {
    int64_t m;
    int64_t k;
    int64_t n;
};

MatmulShape parse_shapes(const std::string& shapes_json) {
    auto nums = extract_numbers(shapes_json);
    if (nums.size() < 4) {
        throw std::runtime_error("shapes must include at least two input shapes");
    }
    int64_t m = nums[0];
    int64_t k = nums[1];
    int64_t k2 = nums[2];
    int64_t n = nums[3];
    if (k != k2) {
        throw std::runtime_error("shape mismatch: input0 k != input1 k");
    }
    if (nums.size() >= 6) {
        int64_t out_m = nums[4];
        int64_t out_n = nums[5];
        if (out_m != m || out_n != n) {
            throw std::runtime_error("output shape does not match matmul result");
        }
    }
    return MatmulShape{m, k, n};
}

template <typename T>
std::vector<T> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }
    file.seekg(0, std::ios::end);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size % static_cast<std::streamsize>(sizeof(T)) != 0) {
        throw std::runtime_error("file size not aligned to dtype for " + path);
    }
    std::vector<T> data(static_cast<size_t>(size) / sizeof(T));
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("failed to read " + path);
    }
    return data;
}

template <typename T>
void write_file(const std::string& path, const std::vector<T>& data) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("failed to open " + path + " for write");
    }
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(T)));
    if (!file) {
        throw std::runtime_error("failed to write " + path);
    }
}

template <typename T>
void run_matmul(const Options& opts, const MatmulShape& shape) {
    const size_t count_a = static_cast<size_t>(shape.m * shape.k);
    const size_t count_b = static_cast<size_t>(shape.k * shape.n);
    auto a = read_file<T>(opts.input0);
    auto b = read_file<T>(opts.input1);
    if (a.size() != count_a || b.size() != count_b) {
        throw std::runtime_error("input sizes do not match shapes");
    }
    std::vector<T> out(static_cast<size_t>(shape.m * shape.n), static_cast<T>(0));
    matmul_kernel<T>(a.data(), b.data(), out.data(), static_cast<size_t>(shape.m), static_cast<size_t>(shape.k),
                     static_cast<size_t>(shape.n));
    write_file<T>(opts.output0, out);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opts = parse_args(argc, argv);
        if (opts.shapes_json.empty()) {
            throw std::runtime_error("--shapes is required");
        }
        MatmulShape shape = parse_shapes(opts.shapes_json);
        if (opts.dtype == "float32") {
            run_matmul<float>(opts, shape);
        } else if (opts.dtype == "int32") {
            run_matmul<int32_t>(opts, shape);
        } else {
            throw std::runtime_error("unsupported dtype: " + opts.dtype);
        }
    } catch (const std::exception& ex) {
        std::cerr << "matmul_runner failed: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
