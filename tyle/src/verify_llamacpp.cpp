// Verification harness: runs llama.cpp's own AMX B-matrix packing code and exposes it to Python.
//
// The packing code (transposes, pack_qs, pack_B, get_tile_size, PACKED_INDEX) is NOT retyped
// here. tools/verify_llamacpp.py cuts it out of mmq_ref.cpp verbatim into
// build/llamacpp_pack_extract.inc, and this file includes that. The AVX2 / AVX-512 intrinsics
// it uses are provided by SIMDe (https://github.com/simd-everywhere/simde) so it also builds
// on CPUs without AVX-512, such as arm64.
//
// Build and run: python tools/verify_llamacpp.py
// Not part of the wheel: setuptools only builds tyle_pack.cpp.

#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx512.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <type_traits>

#define GGML_COMMON_DECL_C
#include "ggml_common_ref.h"

#define RESTRICT __restrict__

// ggml/src/ggml-cpu/amx/common.h
#define TILE_M 16
#define TILE_N 16
#define TILE_K 32
#define VNNI_BLK 4
#define AMX_BLK_SIZE 32

namespace {
#include "llamacpp_pack_extract.inc"

// llama.cpp convert_B_packed_format without parallel_for
template <typename TB, int BLOCK_K>
int ref_convert(uint8_t * packed_B, const uint8_t * B_raw, int N, int K) {
    const TB * B = reinterpret_cast<const TB *>(B_raw);
    const int NB = N / TILE_N;
    const int KB = K / BLOCK_K;
    const int TILE_SIZE = get_tile_size<TB>();
    for (int n = 0; n < NB; ++n) {
        for (int k = 0; k < KB; ++k) {
            int n0 = n * TILE_N;
            pack_B((char *) packed_B + PACKED_INDEX(n, k, KB, TILE_SIZE), &B[n0 * KB + k], KB);
        }
    }
    return TILE_SIZE;
}

template <typename TB>
int ref_tile_size() { return get_tile_size<TB>(); }
}  // namespace

extern "C" {

// Returns llama.cpp's get_tile_size for `quant`, or -1 if unsupported.
int ref_tile_size_of(const char * quant) {
    if (!std::strcmp(quant, "q4_0"))   return ref_tile_size<block_q4_0>();
    if (!std::strcmp(quant, "q4_1"))   return ref_tile_size<block_q4_1>();
    if (!std::strcmp(quant, "q8_0"))   return ref_tile_size<block_q8_0>();
    if (!std::strcmp(quant, "q4_K"))   return ref_tile_size<block_q4_K>();
    if (!std::strcmp(quant, "q5_K"))   return ref_tile_size<block_q5_K>();
    if (!std::strcmp(quant, "q6_K"))   return ref_tile_size<block_q6_K>();
    if (!std::strcmp(quant, "iq4_xs")) return ref_tile_size<block_iq4_xs>();
    return -1;
}

// Packs N rows of raw ggml blocks with llama.cpp's code. `out` must hold NB*KB*tile_size bytes.
// Returns the tile size, or -1 if unsupported.
int ref_pack(const char * quant, const uint8_t * blocks, int N, int K, uint8_t * out) {
    if (!std::strcmp(quant, "q4_0"))   return ref_convert<block_q4_0, QK4_0>(out, blocks, N, K);
    if (!std::strcmp(quant, "q4_1"))   return ref_convert<block_q4_1, QK4_1>(out, blocks, N, K);
    if (!std::strcmp(quant, "q8_0"))   return ref_convert<block_q8_0, QK8_0>(out, blocks, N, K);
    if (!std::strcmp(quant, "q4_K"))   return ref_convert<block_q4_K, QK_K>(out, blocks, N, K);
    if (!std::strcmp(quant, "q5_K"))   return ref_convert<block_q5_K, QK_K>(out, blocks, N, K);
    if (!std::strcmp(quant, "q6_K"))   return ref_convert<block_q6_K, QK_K>(out, blocks, N, K);
    if (!std::strcmp(quant, "iq4_xs")) return ref_convert<block_iq4_xs, QK_K>(out, blocks, N, K);
    return -1;
}

}  // extern "C"
