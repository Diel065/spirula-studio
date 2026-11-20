#include "nn/io/StageDump.h"

#include "core/Env.h"
#include "external/npy.hpp"
#include "nn/core/Error.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace nn {

StageDump::StageDump(const char* env_suffix) {
    const char* d = spirula::env(env_suffix);
    if (!d || !d[0]) return;
    std::error_code ec;
    std::filesystem::create_directories(d, ec);
    dir_ = d;
}

void write_npy_f32(const std::string& path, const float* data,
                   const std::vector<int64_t>& shape) {
    npy::shape_t s;
    for (int64_t d : shape) s.push_back((unsigned long)d);
    npy::npy_data_ptr<float> d{data, s, false};
    std::ofstream out(path, std::ios::binary);
    npy::write_npy(out, d);
    out.close();
    NN_CHECK(!out.fail(), "cannot write %s", path.c_str());
}

void StageDump::host(const char* name, const float* data,
                     const std::vector<int64_t>& shape) const {
    if (dir_.empty()) return;
    write_npy_f32(dir_ + "/" + name + ".npy", data, shape);
}

void StageDump::tensor(const char* name, const Tensor& t,
                       const std::vector<int64_t>& shape) const {
    if (dir_.empty()) return;
    std::vector<float> v((size_t)t.numel());
    tensor_to_host(t, v.data(), t.numel());
    host(name, v.data(), shape);
}

bool read_npy_f32(const std::string& path, std::vector<float>& data,
                  std::vector<int64_t>& shape) {
    try {
        // npy.hpp trusts its input and sizes in 32 bits on Windows: the header and
        // the payload, 4 bytes for each of the shape's elements, are checked first.
        std::ifstream in(path, std::ios::binary);
        unsigned char pre[12];
        if (!in.read((char*)pre, 10) || std::memcmp(pre, "\x93NUMPY", 6) != 0) return false;
        uint64_t len = pre[8] | (uint64_t)pre[9] << 8, start = 10;
        if (pre[6] >= 2) {
            if (!in.read((char*)pre + 10, 2)) return false;
            len |= (uint64_t)pre[10] << 16 | (uint64_t)pre[11] << 24;
            start = 12;
        }
        std::error_code ec;
        const uint64_t file = std::filesystem::file_size(path, ec);
        if (ec || len == 0 || start + len > file) return false;
        std::string header((size_t)len, ' ');
        if (!in.read(&header[0], (std::streamsize)len)) return false;
        const npy::header_t h = npy::parse_header(header);
        uint64_t n = 4;
        for (unsigned long s : h.shape) {
            if (s && n > UINT64_MAX / s) return false;
            n *= s;
        }
        if (h.fortran_order || n != file - start - len) return false;
        in.seekg(0);
        const npy::npy_data<float> d = npy::read_npy<float>(in);
        if (!in || in.peek() != std::char_traits<char>::eof()) return false;
        data = d.data;
        shape.assign(d.shape.begin(), d.shape.end());
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

}  // namespace nn
