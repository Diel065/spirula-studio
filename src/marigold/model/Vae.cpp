#include "marigold/model/Vae.h"

#include "marigold/Common.h"
#include "nn/Ops.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace marigold {
namespace {

using nn::DType;
using nn::Tensor;
using nn::arena_tensor;

// Qwen-Image's VAE: channels at full resolution, 1/2, 1/4 and 1/8.
constexpr int64_t kDims[4] = {96, 192, 384, 384};
// Output positions a streamed step computes. GTX 1660 SUPER, 1536x1152, against
// whole maps: 4096 is 7% slower (encode high water 232 MB), 8192 4% (329 MB), 16384
// 2% (542 MB); more below 1 MP. Bands grow with this and with the width.
constexpr int64_t kStepPositions = 8192;
// The attention's [rows, N] fp32 blocks: scores, and v^T while it is built.
constexpr uint64_t kChunkBytes = 64ull << 20;
// conv2d's column buffer at most (kMaxColBytes, nn/OpConv.cpp).
constexpr uint64_t kColBytes = 32ull << 20;
// A GEMM's kernel, and so its rounding, follows its row count (nn/OpGemm.cpp's
// kBigTileLimit): from this many rows on, it is the one all rows would get.
constexpr int64_t kGemmRows = 128;

}  // namespace

void Vae::load(const nn::SafetensorsFile& f, bool encoder, const char* tag) {
    const std::string keep[2] = {encoder ? "vae.encoder." : "vae.decoder.",
                                 encoder ? "vae.quant_conv." : "vae.post_quant_conv."};
    for (const std::string& name : f.names()) {
        if (name.rfind(keep[0], 0) != 0 && name.rfind(keep[1], 0) != 0) continue;
        nn::OnnxTensor t = f.read(name);
        const bool gamma = name.size() > 6 && name.compare(name.size() - 6, 6, ".gamma") == 0;
        if (gamma) {
            // QwenImageRMS_norm: F.normalize(x) * sqrt(C) * gamma.
            const float s = std::sqrt((float)t.data.size());
            for (float& v : t.data) v *= s;
        }
        w_.stage(name, t.shape, std::move(t.data), t.was_f16);
    }
    w_.upload(tag);
}

void Vae::release() { w_.release(); }

namespace {

void rms(const Tensor& out, const Tensor& x, const Tensor& gamma, bool silu) {
    const int64_t C = x.shape[x.ndim - 1];
    nn::l2_normalize_rows(out.view(out.numel() / C, C), x.view(x.numel() / C, C));
    nn::mul(out, out, gamma, silu ? nn::Act::Silu : nn::Act::None);
}

void conv(vk::Arena& a, const nn::WeightStore& w, const Tensor& out, const Tensor& in,
          const std::string& p, int k, int stride, int pad_y, int pad_x,
          const Tensor& residual = {}) {
    nn::ConvOpts o;
    o.stride_y = o.stride_x = stride;
    o.pad_y = pad_y;
    o.pad_x = pad_x;
    o.bias = w.get(p + ".bias");
    o.residual = residual;
    nn::conv2d(a, out, in, w.get(p + ".weight"), k, k, o);
}

int64_t floor_half(int64_t v) { return v >= 0 ? v / 2 : -((1 - v) / 2); }

// Rows of a [rows, N] fp32 attention block. Past N = 131072 (8.4 MP) the 64 MiB
// would leave fewer than kGemmRows, and the score GEMMs fall to the 64x64 tile.
int64_t attention_chunk(int64_t N) {
    return std::min(N, std::max(kGemmRows, (int64_t)((kChunkBytes / 4) / (uint64_t)N)));
}

// [b0, end) of N: whole chunks, at least kGemmRows rows, and never leaving fewer.
int64_t block_end(int64_t b0, int64_t chunk, int64_t N) {
    int64_t b1 = std::min(N, b0 + chunk);
    while (b1 < N && (b1 - b0 < kGemmRows || N - b1 < kGemmRows)) b1 = std::min(N, b1 + chunk);
    return b1;
}

uint64_t map_bytes(int64_t rows, int64_t W, int64_t C) { return (uint64_t)rows * W * C * 4; }

// Rows [lo, hi) of an H-row map, at rows [lo - base, hi - base) of `buf`. Rows
// -1 and H are held as zeros, so a 3x3 conv over a band needs no row padding
// and sums exactly what it sums over the whole map.
struct Rows {
    Tensor buf;
    int64_t H = 0, base = -1, lo = -1, hi = 0;
    bool sink = false;  // the pass's output: whole, written once, never read

    int64_t room() const { return sink ? H - hi : buf.shape[0] - (hi - lo) - 1; }
    Tensor rows(int64_t a, int64_t b) const {
        NN_CHECK(a >= lo && b <= hi && a < b, "Vae: rows [%lld, %lld) asked of [%lld, %lld)",
                 (long long)a, (long long)b, (long long)lo, (long long)hi);
        return buf.slice0(a - base, b - a);
    }
    void drop(int64_t r) { lo = std::clamp(r, lo, hi); }

    // Rows [hi, hi + n) to write. Moves the held rows to the front when the
    // buffer's tail is short; a copy may not overlap itself.
    Tensor append(vk::Arena& a, int64_t n) {
        NN_CHECK(n > 0 && n <= room(), "Vae: %lld rows into a band with room for %lld",
                 (long long)n, (long long)room());
        if (!sink && hi + n + 1 - base > buf.shape[0]) {
            const int64_t k = hi - lo;
            if (k > 0 && lo - base < k) {
                vk::ArenaScope scope(a);
                const Tensor t = arena_tensor(a, DType::F32, k, buf.shape[1], buf.shape[2]);
                nn::copy(t, rows(lo, hi));
                nn::copy(buf.slice0(0, k), t);
            } else if (k > 0) {
                nn::copy(buf.slice0(0, k), rows(lo, hi));
            }
            base = lo;
        }
        const Tensor out = buf.slice0(hi - base, n);
        hi += n;
        if (!sink && hi == H) {
            nn::fill(buf.slice0(hi - base, 1), 0.0f);
            ++hi;
        }
        return out;
    }
};

Rows band_rows(vk::Arena& a, int64_t H, int64_t W, int64_t C, int64_t cap) {
    Rows r;
    r.buf = arena_tensor(a, DType::F32, cap, W, C);
    r.H = H;
    nn::fill(r.buf.slice0(0, 1), 0.0f);
    return r;
}

// One link of a streamed chain and the map it reads.
struct Stage {
    enum Kind { Res, Up, Down, Conv } kind;
    std::string p, norm;  // weights; Conv's rms + silu gamma, if any
    int64_t H, W, C, cout;
    int64_t band, cap;    // output rows a step; rows its input holds at most
    int64_t done = 0;
    Rows u;               // Res: conv1's output after rms2 + silu
};

// A pass between two whole maps, its intermediates a band of rows at a time:
// each stage pulls the rows it needs from the one before, in bands.
struct Chain {
    int64_t H, W, C;  // the map the next stage added reads
    int64_t positions = kStepPositions;  // per step
    std::vector<Stage> st;
    std::vector<Rows> maps;  // maps[i] feeds st[i]; the last is the sink
    vk::Arena* a = nullptr;
    const nn::WeightStore* w = nullptr;
    Tensor src;
    int64_t fed = 0;

    void add(Stage::Kind k, std::string p, int64_t cout, std::string norm = {}) {
        const int64_t Ho = k == Stage::Up ? 2 * H : k == Stage::Down ? H / 2 : H;
        const int64_t Wo = k == Stage::Up ? 2 * W : k == Stage::Down ? W / 2 : W;
        const int64_t most = std::clamp<int64_t>((positions + Wo - 1) / Wo, 1, Ho);
        const int64_t band = (Ho + (Ho + most - 1) / most - 1) / ((Ho + most - 1) / most);
        // What each step below holds of its input, and the bottom zero row.
        const int64_t cap = k == Stage::Up     ? band / 2 + 5
                            : k == Stage::Down ? 2 * band + 3
                                               : band + 4;
        st.push_back({k, std::move(p), std::move(norm), H, W, C, cout, band, cap});
        H = Ho;
        W = Wo;
        C = cout;
    }

    // The bands, the largest step's temporaries, and a shift's copy.
    uint64_t bytes() const {
        uint64_t held = 0, temp = 0;
        for (size_t i = 0; i < st.size(); ++i) {
            const Stage& s = st[i];
            held += map_bytes(s.cap, s.W, s.C);
            if (s.kind == Stage::Res) held += map_bytes(s.band + 4, s.W, s.cout);
            const uint64_t in = s.kind == Stage::Up ? map_bytes(s.band + 6, 2 * s.W, s.C)
                                                    : map_bytes(s.band + 3, s.W, s.C);
            uint64_t shift = i + 1 < st.size() ? map_bytes(st[i + 1].cap, st[i + 1].W, s.cout) : 0;
            if (s.kind == Stage::Res) shift = std::max(shift, map_bytes(s.band + 4, s.W, s.cout));
            temp = std::max(temp, in + std::max(kColBytes, shift));
        }
        return held + temp;
    }

    void run(vk::Arena& arena, const nn::WeightStore& weights, const Tensor& from,
             const Tensor& to) {
        NN_CHECK(to.shape[0] == H && to.shape[1] == W && to.shape[2] == C,
                 "Vae: a %lldx%lldx%lld pass into a %lldx%lldx%lld map", (long long)H,
                 (long long)W, (long long)C, (long long)to.shape[0], (long long)to.shape[1],
                 (long long)to.shape[2]);
        a = &arena;
        w = &weights;
        src = from;
        for (Stage& s : st) {
            maps.push_back(band_rows(arena, s.H, s.W, s.C, s.cap));
            if (s.kind == Stage::Res) s.u = band_rows(arena, s.H, s.W, s.cout, s.band + 4);
        }
        Rows out;
        out.buf = to;
        out.H = to.shape[0];
        out.base = out.lo = out.hi = 0;
        out.sink = true;
        maps.push_back(out);
        while (maps.back().hi < maps.back().H) step(st.size() - 1);
    }

    // maps[i] holds rows up to r (exclusive; H + 1 is the bottom zero row).
    void need(size_t i, int64_t r) {
        Rows& m = maps[i];
        while (m.hi < r) {
            NN_CHECK(m.room() > 0, "Vae: stage %zu's band is full", i);
            if (i > 0) {
                step(i - 1);
                continue;
            }
            const int64_t n = std::min(m.room(), m.H - fed);
            nn::copy(m.append(*a, n), src.slice0(fed, n));
            fed += n;
        }
    }

    void step(size_t i) {
        Stage& s = st[i];
        Rows& x = maps[i];
        Rows& y = maps[i + 1];
        const int64_t oa = s.done;
        switch (s.kind) {
        case Stage::Res: res(i, s, x, y); return;
        case Stage::Up: {
            // Output row r is the conv over upsampled rows r - 1 .. r + 1.
            const int64_t ob = std::min({oa + s.band, 2 * s.H, oa + y.room()});
            const int64_t ia = floor_half(oa - 1), ib = ob / 2 + 1;
            need(i, ib);
            vk::ArenaScope scope(*a);
            const Tensor up = arena_tensor(*a, DType::F32, 2 * (ib - ia), 2 * s.W, s.C);
            nn::upsample_nearest2x(up, x.rows(ia, ib));
            const Tensor o = y.append(*a, ob - oa);
            conv(*a, *w, o, up.slice0(oa - 1 - 2 * ia, ob - oa + 2), s.p + ".resample.1", 3, 1,
                 0, 1);
            s.done = ob;
            x.drop(floor_half(ob - 1));
            return;
        }
        case Stage::Down: {
            // ZeroPad2d((0, 1, 0, 1)), stride 2: output row r reads rows 2r .. 2r + 2.
            const int64_t ob = std::min({oa + s.band, s.H / 2, oa + y.room()});
            need(i, 2 * ob + 1);
            const Tensor o = y.append(*a, ob - oa);
            conv(*a, *w, o, x.rows(2 * oa, 2 * ob + 1), s.p + ".resample.1", 3, 2, 0, 0);
            s.done = ob;
            x.drop(2 * ob);
            return;
        }
        case Stage::Conv: {
            const int64_t ob = std::min({oa + s.band, s.H, oa + y.room()});
            need(i, ob + 1);
            vk::ArenaScope scope(*a);
            Tensor in = x.rows(oa - 1, ob + 1);
            if (!s.norm.empty()) {
                const Tensor t = arena_tensor(*a, DType::F32, ob - oa + 2, s.W, s.C);
                rms(t, in, w->get(s.norm), true);
                in = t;
            }
            const Tensor o = y.append(*a, ob - oa);
            conv(*a, *w, o, in, s.p, 3, 1, 0, 1);
            s.done = ob;
            x.drop(ob - 1);
            return;
        }
        }
    }

    // x + conv2(silu(rms2(conv1(silu(rms1(x)))))): output row r needs u up to
    // r + 1, which needs x up to r + 2; x row r stays for the sum.
    void res(size_t i, Stage& s, Rows& x, Rows& y) {
        const int64_t oa = s.done, ob = std::min({oa + s.band, s.H, oa + y.room()});
        while (s.u.hi < std::min(ob + 1, s.H + 1)) {
            const int64_t ua = s.u.hi, ub = std::min({ob + 1, s.H, ua + s.u.room()});
            NN_CHECK(ub > ua, "Vae: stage %zu's band is full", i);
            need(i, ub + 1);
            vk::ArenaScope scope(*a);
            const Tensor t = arena_tensor(*a, DType::F32, ub - ua + 2, s.W, s.C);
            rms(t, x.rows(ua - 1, ub + 1), w->get(s.p + ".norm1.gamma"), true);
            const Tensor v = s.u.append(*a, ub - ua);
            conv(*a, *w, v, t, s.p + ".conv1", 3, 1, 0, 1);
            rms(v, v, w->get(s.p + ".norm2.gamma"), true);
        }
        const Tensor o = y.append(*a, ob - oa);
        const Tensor xr = x.rows(oa, ob), ur = s.u.rows(oa - 1, ob + 1);
        if (s.C != s.cout) {
            conv(*a, *w, o, xr, s.p + ".conv_shortcut", 1, 1, 0, 0);
            conv(*a, *w, o, ur, s.p + ".conv2", 3, 1, 0, 1, o);
        } else {
            conv(*a, *w, o, ur, s.p + ".conv2", 3, 1, 0, 1, xr);
        }
        s.done = ob;
        x.drop(ob);
        s.u.drop(ob - 1);
    }
};

std::string down_block(int i) { return "vae.encoder.down_blocks." + std::to_string(i); }

// conv_in, then two blocks a level and a stride-2 conv between levels.
Chain encoder_chain(int64_t H, int64_t W, int64_t step) {
    Chain c{H, W, 3, step};
    c.add(Stage::Conv, "vae.encoder.conv_in", kDims[0]);
    for (int l = 0; l < 4; ++l) {
        for (int r = 0; r < 2; ++r) c.add(Stage::Res, down_block(3 * l + r), kDims[l]);
        if (l < 3) c.add(Stage::Down, down_block(3 * l + 2), kDims[l]);
    }
    return c;
}

// From the mid block's output: three blocks a level, each upsample halving the
// channels, then norm_out and conv_out.
Chain decoder_chain(int64_t h, int64_t w, int64_t step) {
    Chain c{h, w, kDims[3], step};
    for (int u = 0; u < 4; ++u) {
        const std::string p = "vae.decoder.up_blocks." + std::to_string(u);
        for (int r = 0; r < 3; ++r)
            c.add(Stage::Res, p + ".resnets." + std::to_string(r), kDims[3 - u]);
        if (u < 3) c.add(Stage::Up, p + ".upsamplers.0", kDims[3 - u] / 2);
    }
    c.add(Stage::Conv, "vae.decoder.conv_out", 3, "vae.decoder.norm_out.gamma");
    return c;
}

// The attention below at N positions, and a full-size block's two temporaries.
uint64_t mid_bytes(int64_t N) {
    const int64_t C = kDims[3], chunk = attention_chunk(N);
    const uint64_t rows = (uint64_t)std::min(N, chunk + 2 * kGemmRows);
    const uint64_t whole = (uint64_t)N * C * 8;  // fp16 k and v^T, fp32 x normalized
    const uint64_t build = std::max(rows * C, (uint64_t)std::max(std::min(chunk, C), kGemmRows) * N);
    const uint64_t loop = (uint64_t)chunk * N + 2 * rows * C;
    return std::max(whole + std::max(build, loop) * 4, 2 * (uint64_t)N * C * 4 + kColBytes);
}

}  // namespace

// x + conv2(silu(rms2(conv1(silu(rms1(x)))))), the sum written over the
// shortcut, so a block whose width does not change runs in place.
Tensor Vae::res_block(vk::Arena& a, const Tensor& x, int cout, const std::string& p) const {
    const int64_t H = x.shape[0], W = x.shape[1], C = x.shape[2];
    Tensor out = x;
    if (C != cout) {
        out = arena_tensor(a, DType::F32, H, W, cout);
        conv(a, w_, out, x, p + ".conv_shortcut", 1, 1, 0, 0);
    }
    vk::ArenaScope scope(a);
    Tensor t = arena_tensor(a, DType::F32, H, W, C);
    rms(t, x, w_.get(p + ".norm1.gamma"), true);
    Tensor u = arena_tensor(a, DType::F32, H, W, cout);
    conv(a, w_, u, t, p + ".conv1", 3, 1, 1, 1);
    rms(u, u, w_.get(p + ".norm2.gamma"), true);
    conv(a, w_, out, u, p + ".conv2", 3, 1, 1, 1, out);
    return out;
}

// One head as wide as the map (384), past the fused kernel's 256: chunked
// GEMMs and a row softmax instead. softmax(qk) (v + b) = softmax(qk) v + b, so
// v's bias joins at the end. x normalized, fp16 k and v^T are whole; the rest is blocks.
void Vae::attention(vk::Arena& a, const Tensor& x, const std::string& p) const {
    vk::ArenaScope scope(a);
    const int64_t N = x.shape[0] * x.shape[1], C = x.shape[2];
    const Tensor xv = x.view(N, C);
    const Tensor wqkv = w_.get(p + ".to_qkv.weight").asMatrix();
    const Tensor bqkv = w_.get(p + ".to_qkv.bias");
    const int64_t chunk = attention_chunk(N), rows = std::min(N, chunk + 2 * kGemmRows);

    Tensor k16 = arena_tensor(a, DType::F16, N, C);
    Tensor vt16 = arena_tensor(a, DType::F16, C, N);
    Tensor xn = arena_tensor(a, DType::F32, N, C);
    rms(xn, xv, w_.get(p + ".norm.gamma"), false);
    {
        vk::ArenaScope build(a);
        // At least kGemmRows rows, and even: v^T's fp16 blocks start on whole words.
        const int64_t parts = std::min((C + chunk - 1) / chunk, C / kGemmRows);
        const int64_t m = (C + parts - 1) / parts;
        const Tensor f = arena_tensor(a, DType::F32, std::max(rows * C, m * N));
        nn::LinearOpts lo;
        lo.bias = bqkv.offsetElems(C).view(C);
        for (int64_t b0 = 0, b1; b0 < N; b0 = b1) {
            b1 = block_end(b0, chunk, N);
            nn::linear(f.view(b1 - b0, C), xn.slice0(b0, b1 - b0), wqkv.slice0(C, C), lo);
            nn::copy(k16.slice0(b0, b1 - b0), f.view(b1 - b0, C));
        }
        for (int64_t c0 = 0; c0 < C; c0 += m) {
            const int64_t n = std::min(m, C - c0);
            nn::matmul_nt(f.view(n, N), wqkv.slice0(2 * C + c0, n), xn);
            nn::copy(vt16.slice0(c0, n), f.view(n, N));
        }
    }

    Tensor s = arena_tensor(a, DType::F32, chunk, N);
    Tensor q = arena_tensor(a, DType::F32, rows, C);
    Tensor o = arena_tensor(a, DType::F32, rows, C);
    const float scale = 1.0f / std::sqrt((float)C);
    nn::LinearOpts qo, vo, po;
    qo.bias = bqkv.view(C);
    vo.bias = bqkv.offsetElems(2 * C).view(C);
    po.bias = w_.get(p + ".proj.bias");
    const Tensor proj = w_.get(p + ".proj.weight").asMatrix();
    for (int64_t b0 = 0, b1; b0 < N; b0 = b1) {
        b1 = block_end(b0, chunk, N);
        const int64_t nb = b1 - b0;
        const Tensor xr = xv.slice0(b0, nb), qb = q.view(nb, C), ob = o.view(nb, C);
        nn::linear(qb, xn.slice0(b0, nb), wqkv.slice0(0, C), qo);
        for (int64_t q0 = b0; q0 < b1; q0 += chunk) {
            const int64_t n = std::min(chunk, b1 - q0);
            const Tensor sc = s.view(n, N);
            nn::matmul_nt(sc, qb.slice0(q0 - b0, n), k16, scale);
            nn::softmax_rows(sc, sc);
            nn::linear(ob.slice0(q0 - b0, n), sc, vt16, vo);
        }
        po.residual = xr;
        nn::linear(xr, ob, proj, po);
    }
}

void Vae::encode(vk::Arena& a, const Tensor& rgb, const Tensor& mu) const {
    vk::ArenaScope scope(a);
    const int64_t H = rgb.shape[0] / 8, W = rgb.shape[1] / 8;
    const std::string e = "vae.encoder.";
    Chain c = encoder_chain(rgb.shape[0], rgb.shape[1], step_ ? step_ : kStepPositions);
    Tensor x = arena_tensor(a, DType::F32, H, W, c.C);
    {
        vk::ArenaScope bands(a);
        c.run(a, w_, rgb, x);
    }
    const std::string m = e + "mid_block.";
    x = res_block(a, x, (int)x.shape[2], m + "resnets.0");
    attention(a, x, m + "attentions.0");
    x = res_block(a, x, (int)x.shape[2], m + "resnets.1");
    rms(x, x, w_.get(e + "norm_out.gamma"), true);
    Tensor h = arena_tensor(a, DType::F32, H, W, w_.get(e + "conv_out.weight").shape[0]);
    conv(a, w_, h, x, e + "conv_out", 3, 1, 1, 1);
    // quant_conv's first half is the posterior mean; the rest is its log variance.
    nn::LinearOpts lo;
    lo.bias = w_.get("vae.quant_conv.bias").view(mu.shape[2]);
    nn::linear(mu.view(H * W, mu.shape[2]), h.view(H * W, h.shape[2]),
               w_.get("vae.quant_conv.weight").asMatrix().slice0(0, mu.shape[2]), lo);
}

void Vae::decode(vk::Arena& a, const Tensor& z, const Tensor& rgb) const {
    vk::ArenaScope scope(a);
    const int64_t h = z.shape[0], w = z.shape[1];
    const std::string d = "vae.decoder.";
    Tensor zq = arena_tensor(a, DType::F32, h, w, z.shape[2]);
    nn::LinearOpts lo;
    lo.bias = w_.get("vae.post_quant_conv.bias");
    nn::linear(zq.view(h * w, z.shape[2]), z.view(h * w, z.shape[2]),
               w_.get("vae.post_quant_conv.weight").asMatrix(), lo);
    Tensor x = arena_tensor(a, DType::F32, h, w, w_.get(d + "conv_in.weight").shape[0]);
    conv(a, w_, x, zq, d + "conv_in", 3, 1, 1, 1);
    const std::string m = d + "mid_block.";
    x = res_block(a, x, (int)x.shape[2], m + "resnets.0");
    attention(a, x, m + "attentions.0");
    x = res_block(a, x, (int)x.shape[2], m + "resnets.1");
    decoder_chain(h, w, step_ ? step_ : kStepPositions).run(a, w_, x, rgb);
}

// Either pass's peak: the caller's rgb, the 1/8 maps, and the larger of the
// mid block's working set and the streamed bands.
uint64_t Vae::arenaBytes(int H, int W) {
    const int64_t h = H / 8, w = W / 8, N = h * w;
    const uint64_t rgb = (uint64_t)H * W * 3 * 4, map = (uint64_t)N * kDims[3] * 4;
    const uint64_t enc = map + (uint64_t)N * 32 * 4 +
                         std::max(encoder_chain(H, W, kStepPositions).bytes(), mid_bytes(N));
    const uint64_t dec = (uint64_t)N * 16 * 4 + map +
                         std::max(decoder_chain(h, w, kStepPositions).bytes(), mid_bytes(N));
    return rgb + std::max(enc, dec) + (16ull << 20);
}

}  // namespace marigold
