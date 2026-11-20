// Marigold v2: the installer's host arithmetic, the installed files against the
// reference's constants, a prediction with its stages dumped for
// tools/marigold/reference.py, and the VAE alone. Runs what it can, SKIPs the rest.
//
//   marigold_test                           host checks; a small prediction if installed
//   marigold_test --install depth,normals   fetch and convert (~43 GB streamed)
//   marigold_test --constants <ref dir>     installed constants vs a reference run's dumps
//   marigold_test --image IMG|--rgb X.npy [--task T|both|all] [--tokens N] [--repeat N] [--batch N]
//   marigold_test --vae WxH [--image IMG] [--latent Z.npy] [--task T] [--repeat N]
//                 --out DIR keeps the outputs; SS_MARIGOLD_DUMP=<dir> a prediction's stages

#include "marigold/Common.h"
#include "marigold/Marigold.h"
#include "marigold/model/Fetch.h"
#include "marigold/model/HostMath.h"
#include "marigold/model/Install.h"
#include "marigold/model/Layout.h"
#include "marigold/model/Vae.h"
#include "nn/Ops.h"
#include "nn/vk/Context.h"
#include "nn/vk/Memory.h"
#include "nn/vk/Stream.h"
#include "nn/Device.h"
#include "nn/io/Image.h"
#include "nn/io/Resize.h"
#include "nn/io/Safetensors.h"
#include "nn/io/StageDump.h"
#include "core/BufferedFile.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace marigold;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_failures;
}

void test_host_math() {
    std::printf("Host arithmetic\n");
    check(f32_to_bf16(1.0f) == 0x3f80 && bf16r(1.00390625f) == 1.0f &&
              bf16r(1.01171875f) == 1.015625f,
          "bf16 rounds to nearest even");
    // The one timestep: bf16(499) is 500, and 500 / 1000 is exact.
    check(bf16r(bf16r(499.0f) / 1000.0f) == 0.5f, "the timestep is 0.5");

    // NF4: each level decodes to itself times the block's absmax, and an
    // all-zero block stays zero.
    std::vector<uint16_t> w(128);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    for (int i = 0; i < 64; ++i) w[(size_t)i] = f32_to_bf16(nd(rng));
    std::vector<uint8_t> codes(64);
    std::vector<uint16_t> absmax(2);
    nf4_quantize(w.data(), 128, codes.data(), absmax.data());
    std::vector<float> back(128);
    nf4_dequantize(codes.data(), absmax.data(), 128, back.data());
    float am = 0.0f, worst = 0.0f;
    for (int i = 0; i < 64; ++i) am = std::fmax(am, std::fabs(bf16_to_f32(w[(size_t)i])));
    for (int i = 0; i < 64; ++i)
        worst = std::fmax(worst, std::fabs(back[(size_t)i] - bf16_to_f32(w[(size_t)i])));
    check(bf16_to_f32(absmax[0]) == am, "nf4 absmax is the block's max");
    // Half the widest gap between levels (0.161 at the ends) is the worst case.
    check(worst <= 0.0811f * am + 1e-6f, "nf4 error within half a level");
    bool zero = true;
    for (int i = 64; i < 128; ++i) zero = zero && back[(size_t)i] == 0.0f;
    check(zero, "nf4 zero block");
}

// The reference's dump `name` against the installed tensor `ours`. What went
// through bitsandbytes' one-row kernel reproduces bit for bit; what went
// through cuBLAS differs by its reduced-precision reductions, a step or two.
void compare(const nn::SafetensorsFile& f, const std::string& ours, const std::string& ref_dir,
             const std::string& name, double within) {
    std::vector<float> want;
    std::vector<int64_t> shape;
    const std::string path = ref_dir + "/" + name + ".npy";
    if (!std::ifstream(path)) {
        std::printf("  SKIP %s (no %s.npy)\n", ours.c_str(), name.c_str());
        return;
    }
    if (!nn::read_npy_f32(path, want, shape)) {
        check(false, (path + ": not a readable C-order float32 .npy").c_str());
        return;
    }
    const nn::OnnxTensor got = f.read(ours);
    const size_t n = want.size();
    if (n == 0 || got.data.size() < n) {
        check(false, (ours + ": size differs from the reference").c_str());
        return;
    }
    // bf16-rounded constants: count elements more than one bf16 step apart.
    size_t off = 0;
    double sq = 0, err = 0;
    for (size_t i = 0; i < n; ++i) {
        const float a = got.data[i], b = want[i];
        sq += (double)b * b;
        err += (double)(a - b) * (a - b);
        if (std::fabs(a - b) > std::fabs(b) * (1.0f / 128.0f) + 1e-6f) ++off;
    }
    char line[256];
    std::snprintf(line, sizeof line, "%-22s rel L2 %.2e, %zu of %zu past one bf16 step",
                  ours.c_str(), std::sqrt(err / std::fmax(sq, 1e-30)), off, n);
    check((double)off <= (1.0 - within) * (double)n, line);
}

void test_constants(const std::string& ref, Task t) {
    std::printf("Installed constants vs %s\n", ref.c_str());
    if (!base_installed() || !task_installed(t)) {
        std::printf("  SKIP not installed\n");
        return;
    }
    const nn::SafetensorsFile base(base_path()), task(task_path(t));
    compare(base, "temb", ref, "temb", 0.999);
    std::vector<float> mods;
    std::vector<int64_t> shape;
    for (const char* s : {"img", "txt"}) {
        const std::string path = ref + "/mod_" + s + ".npy";
        if (!std::ifstream(path)) continue;
        if (!nn::read_npy_f32(path, mods, shape)) {
            check(false, (path + ": not a readable C-order float32 .npy").c_str());
            continue;
        }
        size_t off = 0;
        bool whole = true;
        for (int i = 0; i < kLayers && whole; ++i) {
            const nn::OnnxTensor got =
                base.read(block_tensor(i, s[0] == 'i' ? "mod_img" : "mod_txt"));
            whole = mods.size() >= (size_t)(i + 1) * got.data.size();
            for (size_t k = 0; whole && k < got.data.size(); ++k) {
                const float b = mods[(size_t)i * got.data.size() + k];
                if (std::fabs(got.data[k] - b) > std::fabs(b) / 128.0f + 1e-6f) ++off;
            }
        }
        char line[128];
        std::snprintf(line, sizeof line, "mod_%s: %zu of %zu past one bf16 step", s, off,
                      mods.size());
        check(whole && off <= mods.size() / 100, line);
    }
    compare(task, "txt", ref, "txt0", 0.95);
    compare(task, "norm_out", ref, "norm_out", 0.95);
}

// Portable float map, bottom row first: what pbrt-v4 and HDR tools read.
void save_pfm(const std::string& path, const std::vector<float>& v, int w, int h, int c) {
    spirula::BufferedOutputFile f(path, std::ios::binary);
    f << (c == 3 ? "PF" : "Pf") << "\n" << w << " " << h << "\n-1.0\n";
    for (int y = h - 1; y >= 0; --y)
        f.write(reinterpret_cast<const char*>(&v[(size_t)y * w * c]), (std::streamsize)w * c * 4);
    f.close();
    NN_CHECK(!f.fail(), "cannot write %s", path.c_str());
}

void save_png(const nn::Image& img, const std::string& path) {
    NN_CHECK(nn::save_image(img, path, -1), "cannot write %s", path.c_str());
}

float srgb_encode(float v) {
    v = std::clamp(v, 0.0f, 1.0f);
    return v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
}

void save_outputs(const Prediction& p, const std::string& dir) {
    if (!p.depth_log.empty()) save_pfm(dir + "/depth_log.pfm", p.depth_log, p.width, p.height, 1);
    if (!p.normal.empty()) save_pfm(dir + "/normal.pfm", p.normal, p.width, p.height, 3);
    if (!p.albedo.empty()) {
        save_pfm(dir + "/albedo_linear.pfm", p.albedo, p.width, p.height, 3);
        nn::Image img;
        img.width = p.width;
        img.height = p.height;
        img.data.resize(p.albedo.size());
        for (size_t i = 0; i < p.albedo.size(); ++i)
            img.data[i] = (uint8_t)std::lround(255.0f * srgb_encode(p.albedo[i]));
        save_png(img, dir + "/albedo_srgb.png");
    }
    if (!p.depth_log.empty()) {
        float lo = 1e9f, hi = -1e9f;
        for (float v : p.depth_log) lo = std::fmin(lo, v), hi = std::fmax(hi, v);
        nn::Image img;
        img.width = p.width;
        img.height = p.height;
        img.data.resize((size_t)p.width * p.height * 3);
        for (size_t i = 0; i < p.depth_log.size(); ++i)
            img.data[i * 3] = img.data[i * 3 + 1] = img.data[i * 3 + 2] =
                (uint8_t)std::lround(255.0f * (p.depth_log[i] - lo) / std::fmax(hi - lo, 1e-6f));
        save_png(img, dir + "/depth.png");
    }
    if (!p.normal.empty()) {
        nn::Image img;
        img.width = p.width;
        img.height = p.height;
        img.data.resize(p.normal.size());
        // The usual picture of OpenCV normals: x right red, y up green, z out blue.
        for (size_t i = 0; i * 3 < p.normal.size(); ++i) {
            const float v[3] = {p.normal[i * 3], -p.normal[i * 3 + 1], -p.normal[i * 3 + 2]};
            for (int c = 0; c < 3; ++c)
                img.data[i * 3 + c] = (uint8_t)std::lround(127.5f + 127.5f * v[c]);
        }
        save_png(img, dir + "/normal.png");
    }
}

int run_prediction(const std::string& image, const std::string& npy, const std::string& tasks,
                   int tokens, int repeat, int batch, const std::string& out,
                   bool synthetic = false) {
    std::printf("Prediction\n");
    const bool d = tasks.find("depth") != std::string::npos || tasks == "both" || tasks == "all";
    const bool n = tasks.find("normal") != std::string::npos || tasks == "both" || tasks == "all";
    const bool al = tasks.find("albedo") != std::string::npos || tasks == "all";
    if (!base_installed() || (d && !task_installed(Task::Depth)) ||
        (n && !task_installed(Task::Normals)) || (al && !task_installed(Task::Albedo))) {
        std::printf("  SKIP not installed\n");
        return 0;
    }
    int w = 0, h = 0;
    std::vector<float> rgb;
    if (synthetic) {
        // A lit box in a corridor: enough structure for a finite, oriented answer.
        w = 256;
        h = 192;
        rgb.resize((size_t)w * h * 3);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const bool box = x > 96 && x < 160 && y > 80 && y < 150;
                const float v = box ? 0.8f : 0.2f + 0.6f * (float)y / (float)h;
                for (int c = 0; c < 3; ++c)
                    rgb[((size_t)y * w + x) * 3 + c] = v * (0.8f + 0.1f * (float)c);
            }
    } else if (!npy.empty()) {
        std::vector<int64_t> shape;
        NN_CHECK(nn::read_npy_f32(npy, rgb, shape) && shape.size() == 3 && shape[2] == 3,
                 "%s: not an [H, W, 3] float32 array", npy.c_str());
        h = (int)shape[0];
        w = (int)shape[1];
    } else {
        const nn::Image img = nn::load_image(image);
        NN_CHECK(!img.empty(), "cannot read %s", image.c_str());
        w = img.width;
        h = img.height;
        rgb.resize(img.data.size());
        for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = img.data[i] / 255.0f;
    }
    Predictor pred;
    double t0 = now_ms();
    pred.load(d, n, al);
    std::printf("  loaded in %.1f s\n", (now_ms() - t0) / 1000.0);
    PredictOptions o;
    o.want_depth = d;
    o.want_normal = n;
    o.want_albedo = al;
    o.num_tokens = tokens;
    int pw, ph;
    Predictor::processingSize(w, h, tokens, &pw, &ph);
    std::printf("  %dx%d runs at %dx%d (%d tokens), batch capacity %d\n", w, h, pw, ph,
                (pw / kPatch) * (ph / kPatch), pred.batchCapacity(w, h, tokens));
    std::vector<Prediction> p;
    // A batch of mirror images: the same work as distinct ones, and image 0
    // must come back exactly as it does alone.
    std::vector<float> flipped(rgb.size());
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c)
                flipped[((size_t)y * w + x) * 3 + c] = rgb[((size_t)y * w + (w - 1 - x)) * 3 + c];
    std::vector<Predictor::Image> images{{rgb.data(), w, h}};
    for (int b = 1; b < batch; ++b) images.push_back({b % 2 ? flipped.data() : rgb.data(), w, h});
    for (int r = 0; r < repeat; ++r) {
        t0 = now_ms();
        p = pred.predict(images, o);
        std::printf("  predict %.2f s (%.2f s an image)\n", (now_ms() - t0) / 1000.0,
                    (now_ms() - t0) / 1000.0 / batch);
    }
    if (batch > 1) {
        const std::vector<Prediction> one = pred.predict({{rgb.data(), w, h}}, o);
        check(one[0].depth_log == p[0].depth_log && one[0].normal == p[0].normal &&
                  one[0].albedo == p[0].albedo && p[2 % batch].depth_log == p[0].depth_log,
              "a batch gives each image what it gets alone");
    }
    std::printf("  device %.2f GB, peak allocation %.2f GB\n", pred.deviceBytes() / 1e9,
                vk::Allocator::get().peakBytes() / 1e9);
    vk::Stream::get().report();
    if (!out.empty()) save_outputs(p[0], out);
    bool finite = true;
    for (float v : p[0].depth_log) finite = finite && std::isfinite(v);
    for (float v : p[0].normal) finite = finite && std::isfinite(v);
    for (float v : p[0].albedo) finite = finite && std::isfinite(v);
    check(finite, "prediction is finite");
    return 0;
}

// [0, 1] RGB: a ramp, a zone plate reaching Nyquist, 1- and 2-pixel checkers, a
// hard-edged disc and hashed noise, the structures a strip boundary could disturb.
std::vector<float> vae_pattern(int W, int H) {
    std::vector<float> v((size_t)W * H * 3);
    const float k = 3.14159265f / (float)W;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float* p = &v[((size_t)y * W + x) * 3];
            const bool right = 2 * x >= W, low = 2 * y >= H;
            float c[3] = {0.15f + 0.7f * x / W, 0.15f + 0.7f * y / H, 0.5f};
            if (right && !low) {
                const float dx = x - 0.75f * W, dy = y - 0.25f * H;
                c[0] = c[1] = c[2] = 0.5f + 0.45f * std::cos(k * (dx * dx + dy * dy));
            } else if (!right && low) {
                const int s = 4 * y >= 3 * H ? 0 : 1;
                c[0] = c[1] = c[2] = (((x >> s) + (y >> s)) & 1) ? 0.9f : 0.1f;
            } else if (right && low) {
                const float dx = x - 0.75f * W, dy = y - 0.75f * H, r = 0.2f * std::min(W, H);
                if (dx * dx + dy * dy < r * r) c[0] = 0.95f, c[1] = 0.3f, c[2] = 0.05f;
            }
            uint32_t s = (uint32_t)y * 73856093u ^ (uint32_t)x * 19349663u;
            for (int i = 0; i < 3; ++i) {
                s = s * 1664525u + 1013904223u;
                p[i] = std::clamp(c[i] + 0.04f * ((float)(s >> 8) / 16777216.0f - 0.5f), 0.0f, 1.0f);
            }
        }
    return v;
}

double sync_ms() {
    vk::Stream::get().sync();
    return now_ms();
}

// Encode and decode a pattern at the default band size and at one small enough
// to stream every level in many steps: streaming must not change a bit.
void test_vae_bands() {
    std::printf("VAE bands\n");
    if (!base_installed() || !task_installed(Task::Depth)) {
        std::printf("  SKIP not installed\n");
        return;
    }
    const int W = 320, H = 240, h = H / 8, w = W / 8;
    const nn::SafetensorsFile base(base_path()), tf(task_path(Task::Depth));
    Vae enc, dec;
    enc.load(base, true, "marigold.encoder");
    dec.load(tf, false, "marigold.decoder");
    const std::vector<float> rgb = vae_pattern(W, H);
    vk::Arena a{"vae bands"};
    // Tensor cores and MPSGraph take only GEMMs of 128 rows or more, and round
    // differently: off, so a band's row count cannot change the arithmetic.
    struct FixedGemm {
        bool coop = nn::coop_matrix_enabled(), metal = nn::metal_enabled();
        FixedGemm() { nn::set_coop_matrix_enabled(false), nn::set_metal_enabled(false); }
        ~FixedGemm() { nn::set_coop_matrix_enabled(coop), nn::set_metal_enabled(metal); }
    } fixed;
    a.reserve(Vae::arenaBytes(H, W) + (uint64_t)h * w * kLatent * 4 + (2ull << 20));
    std::vector<float> out[2];
    for (int k = 0; k < 2; ++k) {
        enc.setStep(k ? 512 : 0);
        dec.setStep(k ? 512 : 0);
        vk::ArenaScope scope(a);
        nn::Tensor x = nn::arena_tensor(a, nn::DType::F32, H, W, 3);
        nn::Tensor mu = nn::arena_tensor(a, nn::DType::F32, h, w, kLatent);
        nn::tensor_from_host(x, rgb.data(), x.numel());
        nn::unary(x, x, nn::Act::None, 2.0f, -1.0f);
        enc.encode(a, x, mu);
        dec.decode(a, mu, x);
        out[k].resize((size_t)x.numel() + mu.numel());
        nn::tensor_to_host(mu, out[k].data(), mu.numel());
        nn::tensor_to_host(x, out[k].data() + mu.numel(), x.numel());
    }
    check(std::memcmp(out[0].data(), out[1].data(), out[0].size() * 4) == 0,
          "the streamed VAE gives the same bits at another band size");
}

// The VAE alone: encode, then decode the posterior mean (or --latent) with the
// task's decoder. Each pass gets its own arena, so each high water is its own.
int run_vae(int W, int H, const std::string& image, const std::string& latent, Task task,
            int repeat, const std::string& out) {
    std::printf("VAE\n");
    if (!base_installed() || !task_installed(task)) {
        std::printf("  SKIP not installed\n");
        return 0;
    }
    NN_CHECK(W % 16 == 0 && H % 16 == 0 && W > 0 && H > 0, "--vae: W and H must be multiples of 16");
    std::vector<float> rgb;
    if (image.empty()) {
        rgb = vae_pattern(W, H);
    } else {
        const nn::Image img = nn::load_image(image);
        NN_CHECK(!img.empty(), "cannot read %s", image.c_str());
        std::vector<float> f(img.data.size());
        for (size_t i = 0; i < f.size(); ++i) f[i] = img.data[i] / 255.0f;
        rgb = img.width == W && img.height == H
                  ? f
                  : nn::resize_rgb_bicubic(f.data(), img.width, img.height, W, H);
    }
    const int h = H / 8, w = W / 8;
    const vk::Context& ctx = vk::Context::get();
    const uint64_t dev0 = ctx.memoryBudget().usage_bytes;
    const nn::SafetensorsFile base(base_path()), tf(task_path(task));
    Vae enc, dec;
    enc.load(base, true, "marigold.encoder");
    dec.load(tf, false, "marigold.decoder");
    const uint64_t dev1 = ctx.memoryBudget().usage_bytes;
    const uint64_t lat_bytes = (uint64_t)h * w * kLatent * 4;
    const uint64_t reserve = Vae::arenaBytes(H, W) + lat_bytes + (2ull << 20);
    std::printf("  %dx%d, latent %dx%d; weights %.1f MB encoder, %.1f MB decoder; arena %.1f MB\n",
                W, H, w, h, enc.deviceBytes() / 1e6, dec.deviceBytes() / 1e6, reserve / 1e6);

    std::vector<double> te, td;
    std::vector<float> mu_host((size_t)h * w * kLatent);
    uint64_t enc_hw = 0, dec_hw = 0, dev_enc = 0, dev_dec = 0;
    {
        vk::Arena a{"vae bench"};
        a.reserve(reserve);
        dev_enc = ctx.memoryBudget().usage_bytes;
        nn::Tensor x = nn::arena_tensor(a, nn::DType::F32, H, W, 3);
        nn::Tensor mu = nn::arena_tensor(a, nn::DType::F32, h, w, kLatent);
        nn::tensor_from_host(x, rgb.data(), x.numel());
        nn::unary(x, x, nn::Act::None, 2.0f, -1.0f);
        const uint64_t io = a.mark();
        for (int r = 0; r < repeat; ++r) {
            const double t0 = sync_ms();
            enc.encode(a, x, mu);
            te.push_back(sync_ms() - t0);
        }
        enc_hw = a.highWater() - io;
        nn::tensor_to_host(mu, mu_host.data(), mu.numel());
    }
    std::vector<float> z_host = mu_host;
    if (!latent.empty()) {
        std::vector<int64_t> shape;
        NN_CHECK(nn::read_npy_f32(latent, z_host, shape) && shape.size() == 3 && shape[0] == h &&
                     shape[1] == w && shape[2] == kLatent,
                 "%s: not a [%d, %d, %d] float32 array", latent.c_str(), h, w, kLatent);
    }
    std::vector<float> dec_host((size_t)W * H * 3);
    {
        vk::Arena a{"vae bench"};
        a.reserve(reserve);
        dev_dec = ctx.memoryBudget().usage_bytes;
        nn::Tensor z = nn::arena_tensor(a, nn::DType::F32, h, w, kLatent);
        nn::Tensor y = nn::arena_tensor(a, nn::DType::F32, H, W, 3);
        nn::tensor_from_host(z, z_host.data(), z.numel());
        const uint64_t io = a.mark();
        for (int r = 0; r < repeat; ++r) {
            const double t0 = sync_ms();
            dec.decode(a, z, y);
            td.push_back(sync_ms() - t0);
        }
        dec_hw = a.highWater() - io;
        nn::tensor_to_host(y, dec_host.data(), y.numel());
    }
    auto list = [](const std::vector<double>& t) {
        std::string s;
        for (double v : t) s += (s.empty() ? "" : ",") + std::to_string(v);
        return s;
    };
    bool finite = true;
    for (float v : mu_host) finite = finite && std::isfinite(v);
    for (float v : dec_host) finite = finite && std::isfinite(v);
    std::printf("  encode %.1f ms (first %.1f), decode %.1f ms (first %.1f)\n", te.back(), te[0],
                td.back(), td[0]);
    std::printf("  high water: encode %.1f MB, decode %.1f MB (%.0f and %.0f B a pixel)\n",
                enc_hw / 1e6, dec_hw / 1e6, (double)enc_hw / ((double)W * H),
                (double)dec_hw / ((double)W * H));
    std::printf("VAEBENCH {\"W\":%d,\"H\":%d,\"task\":\"%s\",\"enc_ms\":[%s],\"dec_ms\":[%s],"
                "\"enc_hw\":%llu,\"dec_hw\":%llu,\"reserve\":%llu,\"alloc_peak\":%llu,"
                "\"dev_before_load\":%llu,\"dev_loaded\":%llu,\"dev_enc\":%llu,\"dev_dec\":%llu}\n",
                W, H, task_name(task), list(te).c_str(),
                list(td).c_str(), (unsigned long long)enc_hw, (unsigned long long)dec_hw,
                (unsigned long long)reserve,
                (unsigned long long)vk::Allocator::get().peakBytes(), (unsigned long long)dev0,
                (unsigned long long)dev1, (unsigned long long)dev_enc,
                (unsigned long long)dev_dec);
    vk::Stream::get().report();
    if (!out.empty()) {
        nn::write_npy_f32(out + "/mu.npy", mu_host.data(), {h, w, kLatent});
        nn::write_npy_f32(out + "/dec.npy", dec_host.data(), {H, W, 3});
    }
    check(finite, "VAE output is finite");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    nn::set_log_level(2);
    std::string to_install, constants, image, npy, latent, tasks = "both", out;
    int tokens = 3600, repeat = 1, batch = 1, vae_w = 0, vae_h = 0;
    Task task = Task::Depth;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--install" && i + 1 < argc) to_install = argv[++i];
        else if (a == "--constants" && i + 1 < argc) constants = argv[++i];
        else if (a == "--task" && i + 1 < argc) {
            tasks = argv[++i];
            task = tasks == "normals" ? Task::Normals
                   : tasks == "albedo" ? Task::Albedo
                                       : Task::Depth;
        }
        else if (a == "--image" && i + 1 < argc) image = argv[++i];
        else if (a == "--rgb" && i + 1 < argc) npy = argv[++i];
        else if (a == "--tokens" && i + 1 < argc) tokens = std::atoi(argv[++i]);
        else if (a == "--repeat" && i + 1 < argc) repeat = std::atoi(argv[++i]);
        else if (a == "--batch" && i + 1 < argc) batch = std::max(1, std::atoi(argv[++i]));
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--latent" && i + 1 < argc) latent = argv[++i];
        else if (a == "--vae" && i + 1 < argc)
            std::sscanf(argv[++i], "%dx%d", &vae_w, &vae_h);
    }
    try {
        if (!to_install.empty()) {
            const bool d = to_install.find("depth") != std::string::npos;
            const bool n = to_install.find("normal") != std::string::npos;
            const bool al = to_install.find("albedo") != std::string::npos;
            const InstallNeeds need = install_needs(d, n, al);
            std::printf("to stream %.2f GB, to write %.2f GB, %.1f GB free\n",
                        need.download / 1e9, need.disk / 1e9, need.free / 1e9);
            const double t0 = now_ms();
            install(d, n, [&](uint64_t done, uint64_t total) {
                static double last = 0;
                const double now = now_ms();
                if (now - last < 2000 && done < total) return;
                last = now;
                std::printf("\r  %.2f / %.2f GB  %.1f MB/s   ", done / 1e9, total / 1e9,
                            done / 1e3 / std::fmax(now - t0, 1.0));
                std::fflush(stdout);
            }, nullptr, al);
            std::printf("\ninstalled in %.0f s\n", (now_ms() - t0) / 1000.0);
            return 0;
        }
        test_host_math();
        if (!constants.empty()) test_constants(constants, task);
        if (vae_w > 0)
            run_vae(vae_w, vae_h, image, latent, task, std::max(1, repeat), out);
        else if (!image.empty() || !npy.empty())
            run_prediction(image, npy, tasks, tokens, repeat, batch, out);
        else if (constants.empty()) {
            test_vae_bands();
            run_prediction("", "", "both", 192, 1, 2, "", true);
        }
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        nn::shutdown();
        return 1;
    }
    nn::shutdown();
    std::printf("%s\n", g_failures ? "FAILURES" : "all passed");
    return g_failures ? 1 : 0;
}
