#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>

namespace {

struct Options {
    std::string dtype = "float32";
    std::string input0 = "input/input0.bin";
    std::string input1 = "input/input1.bin";
    std::string output = "output/output0.bin";
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
        } else if (arg == "--output" && i + 1 < argc) {
            opt.output = argv[++i];
        }
    }
    return opt;
}

template <typename T>
std::vector<T> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
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
std::vector<T> add_vectors(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("input sizes differ");
    }
    std::vector<T> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = static_cast<T>(a[i] + b[i]);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opts = parse_args(argc, argv);
        std::filesystem::create_directories(std::filesystem::path(opts.output).parent_path());
        if (opts.dtype == "float32") {
            auto a = read_file<float>(opts.input0);
            auto b = read_file<float>(opts.input1);
            auto out = add_vectors<float>(a, b);
            write_file<float>(opts.output, out);
        } else if (opts.dtype == "int32") {
            auto a = read_file<int32_t>(opts.input0);
            auto b = read_file<int32_t>(opts.input1);
            auto out = add_vectors<int32_t>(a, b);
            write_file<int32_t>(opts.output, out);
        } else {
            throw std::runtime_error("unsupported dtype: " + opts.dtype);
        }
    } catch (const std::exception& ex) {
        std::cerr << "add_custom failed: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
