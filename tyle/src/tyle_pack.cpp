// TYLE AMX weight packing.
//
// Portable scalar port of the B-matrix packing in llama.cpp
// ggml/src/ggml-cpu/amx/mmq.cpp (see mmq_ref.cpp next to this file):
//   pack_qs<...>, pack_B(...), s8s8_compensation, convert_B_packed_format.
// The AVX2 / AVX-512 transposes are replaced by plain loops that produce the same bytes,
// so it builds and runs on any CPU (including arm64). Input is raw ggml blocks, N rows of
// K/block_k blocks, exactly as stored in a GGUF tensor. Output is the {NB, KB, TILE_SIZE}
// grid that llama.cpp's AMX kernels consume (PACKED_INDEX = (n * KB + k) * TILE_SIZE).
//
// Exposed to Python as tyle._amxpack:
//   quants() -> tuple[str]
//   info(quant) -> (block_k, block_bytes, tile_bytes)
//   pack(quant, blocks, n, k) -> bytes

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <cstring>

#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "TYLE is little-endian only"
#endif

namespace {

constexpr int TILE_N = 16;
constexpr int TILE_K = 32;
constexpr int QK_K = 256;

// ggml block layouts (ggml-common.h). ggml_half is stored as raw uint16 bits.
struct block_q4_0   { uint16_t d; uint8_t qs[16]; };
struct block_q4_1   { uint16_t d; uint16_t m; uint8_t qs[16]; };
struct block_q8_0   { uint16_t d; int8_t qs[32]; };
struct block_q4_K   { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qs[128]; };
struct block_q5_K   { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
struct block_q6_K   { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };
struct block_iq4_xs { uint16_t d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; };

static_assert(sizeof(block_q4_0) == 18, "q4_0 block size");
static_assert(sizeof(block_q4_1) == 20, "q4_1 block size");
static_assert(sizeof(block_q8_0) == 34, "q8_0 block size");
static_assert(sizeof(block_q4_K) == 144, "q4_K block size");
static_assert(sizeof(block_q5_K) == 176, "q5_K block size");
static_assert(sizeof(block_q6_K) == 210, "q6_K block size");
static_assert(sizeof(block_iq4_xs) == 136, "iq4_xs block size");

inline void put_u16(uint8_t * p, uint16_t v) { std::memcpy(p, &v, 2); }

// In-place transpose of a 16x16 matrix of 32-bit elements; row i is v[i][0..63].
// (llama.cpp transpose_16x16_32bit)
inline void transpose_16x16_32bit(uint8_t (*v)[64]) {
    for (int i = 0; i < 16; ++i) {
        for (int j = i + 1; j < 16; ++j) {
            uint8_t t[4];
            std::memcpy(t, &v[i][4 * j], 4);
            std::memcpy(&v[i][4 * j], &v[j][4 * i], 4);
            std::memcpy(&v[j][4 * i], t, 4);
        }
    }
}

// Result of the two transpose_8x8_32bit calls in pack_qs: 16 rows of 32 values become
// 8 k-groups of {16 rows x 4 values}.  out[j*64 + n*4 + b] = rows[n][4*j + b]
inline void interleave_rows32(const uint8_t rows[16][32], uint8_t * out) {
    for (int j = 0; j < 8; ++j) {
        for (int n = 0; n < 16; ++n) {
            std::memcpy(out + j * 64 + n * 4, rows[n] + 4 * j, 4);
        }
    }
}

// ---- q8_0 ---------------------------------------------------------------------------
// packed_B layout: quants {TILE_N, TILE_K} int8 (VNNI interleaved), d0 {TILE_N} half,
//                  comp {TILE_N} int32 (s8s8 compensation)
void pack_B(uint8_t * out, const block_q8_0 * b) {
    uint8_t rows[16][32];
    for (int n = 0; n < TILE_N; ++n) std::memcpy(rows[n], b[n].qs, 32);
    interleave_rows32(rows, out);
    for (int n = 0; n < TILE_N; ++n) put_u16(out + TILE_N * TILE_K + 2 * n, b[n].d);
    // _mm512_dpbusd_epi32(comp, 0x80 bytes, quants): 128 * sum of the column's 32 quants
    for (int n = 0; n < TILE_N; ++n) {
        int32_t sum = 0;
        for (int i = 0; i < 32; ++i) sum += b[n].qs[i];
        const int32_t comp = 128 * sum;
        std::memcpy(out + TILE_N * TILE_K + TILE_N * 2 + 4 * n, &comp, 4);
    }
}

// ---- q4_0 / q4_1 --------------------------------------------------------------------
// generic pack_qs: nibbles unpacked to bytes, interleaved like q8_0, then two k-groups
// are packed into one byte (low nibble = even group).
template <typename TB>
void pack_qs_q4(uint8_t * out, const TB * b) {
    uint8_t rows[16][32];
    uint8_t tmp[512];
    for (int n = 0; n < TILE_N; ++n) {
        for (int i = 0; i < 16; ++i) {
            rows[n][i]      = b[n].qs[i] & 0xF;
            rows[n][16 + i] = b[n].qs[i] >> 4;
        }
    }
    interleave_rows32(rows, tmp);
    for (int m = 0; m < 4; ++m) {
        for (int x = 0; x < 64; ++x) {
            out[m * 64 + x] = tmp[(2 * m) * 64 + x] | (tmp[(2 * m + 1) * 64 + x] << 4);
        }
    }
}

// packed_B layout: quants {TILE_N, TILE_K/2} uint8, d0 {TILE_N} half
void pack_B(uint8_t * out, const block_q4_0 * b) {
    pack_qs_q4(out, b);
    for (int n = 0; n < TILE_N; ++n) put_u16(out + TILE_N * TILE_K / 2 + 2 * n, b[n].d);
}

// packed_B layout: quants {TILE_N, TILE_K/2} uint8, d0 {TILE_N} half, m0 {TILE_N} half
void pack_B(uint8_t * out, const block_q4_1 * b) {
    pack_qs_q4(out, b);
    for (int n = 0; n < TILE_N; ++n) {
        put_u16(out + TILE_N * TILE_K / 2 + 2 * n, b[n].d);
        put_u16(out + TILE_N * TILE_K / 2 + 2 * TILE_N + 2 * n, b[n].m);
    }
}

// ---- K-quants -----------------------------------------------------------------------
// convert 8 * {min, scale} from int6 to int8
inline void unpack_mins_and_scales(const uint8_t * scales, uint32_t * utmp) {
    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;
    std::memcpy(utmp, scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
}

template <typename TB>
void pack_scales_mins(uint8_t * scales, const TB * b) {
    uint8_t * mins = scales + 8 * TILE_N;
    uint8_t * d = mins + 8 * TILE_N;
    uint8_t * dmin = d + 2 * TILE_N;
    for (int n = 0; n < TILE_N; ++n) {
        uint32_t u32[4];
        uint8_t u8[16];
        unpack_mins_and_scales(b[n].scales, u32);
        std::memcpy(u8, u32, 16);
        for (int k = 0; k < 8; ++k) {
            scales[k * TILE_N + n] = u8[k];
            mins[(k >> 1) * TILE_N * 2 + n * 2 + (k & 1)] = u8[k + 8];
        }
        put_u16(d + 2 * n, b[n].d);
        put_u16(dmin + 2 * n, b[n].dmin);
    }
}

// packed_B layout: quants {8, TILE_N, 16} uint8, scales {8, TILE_N} uint8, mins {8, TILE_N} uint8,
//                  d {TILE_N} half, dmin {TILE_N} half
void pack_B(uint8_t * out, const block_q4_K * b) {
    uint8_t v[16][64];
    uint8_t * pb = out;
    for (int k = 0; k < QK_K / 64; ++k) {  // 2 groups of 32 per step
        for (int n = 0; n < TILE_N; ++n) {
            for (int p = 0; p < 32; ++p) {
                const uint8_t q = b[n].qs[k * 32 + p];
                v[n][p]      = q & 0xF;
                v[n][32 + p] = q >> 4;
            }
        }
        transpose_16x16_32bit(v);
        for (int n = 0; n < TILE_N; n += 2) {
            for (int x = 0; x < 64; ++x) pb[x] = v[n][x] | (v[n + 1][x] << 4);
            pb += 64;
        }
    }
    pack_scales_mins(out + (QK_K / 2) * TILE_N, b);
}

// packed_B layout: quants {8, TILE_N, 16} uint8, qh {8, TILE_N, 4} uint8, scales {8, TILE_N} uint8,
//                  mins {8, TILE_N} uint8, d {TILE_N} half, dmin {TILE_N} half
void pack_B(uint8_t * out, const block_q5_K * b) {
    uint8_t v[16][64];
    uint8_t * pb = out;
    uint8_t * ph = out + (QK_K / 2) * TILE_N;
    for (int k = 0; k < QK_K / 64; ++k) {
        for (int n = 0; n < TILE_N; ++n) {
            for (int p = 0; p < 32; ++p) {
                const uint8_t q = b[n].qs[k * 32 + p];
                const uint8_t h = b[n].qh[p];
                v[n][p]      = (q & 0xF)        | (((h >> (2 * k))     & 1) << 4);
                v[n][32 + p] = ((q >> 4) & 0xF) | (((h >> (2 * k + 1)) & 1) << 4);
            }
        }
        transpose_16x16_32bit(v);
        // 1. lower 4 bits of 2 groups
        for (int n = 0; n < TILE_N; n += 2) {
            for (int x = 0; x < 64; ++x) pb[x] = (v[n][x] & 0xF) | ((v[n + 1][x] & 0xF) << 4);
            pb += 64;
        }
        // 2. higher 1 bit of 2 groups
        for (int g = 0; g < 2; ++g) {
            for (int x = 0; x < 64; ++x) {
                uint8_t hb = 0;
                for (int j = 0; j < 8; ++j) hb |= ((v[g * 8 + j][x] >> 4) & 1) << j;
                ph[x] = hb;
            }
            ph += 64;
        }
    }
    pack_scales_mins(out + (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N, b);
}

// packed_B layout: quants {16, TILE_N, 8} uint8, qh {16, TILE_N, 4} uint8, scales {16, TILE_N} uint8,
//                  d {TILE_N} half
void pack_B(uint8_t * out, const block_q6_K * b) {
    uint8_t v[32][64];
    uint8_t * pb = out;
    uint8_t * ph = out + (QK_K / 2) * TILE_N;
    for (int k = 0; k < QK_K / 128; ++k) {  // 4 groups of 32 per step
        for (int n = 0; n < TILE_N; ++n) {
            const uint8_t * ql = b[n].ql + k * 64;
            const uint8_t * qh = b[n].qh + k * 32;
            for (int p = 0; p < 32; ++p) {
                const uint8_t h = qh[p];
                v[n][p]           = (ql[p] & 0xF)      | (((h >> 0) & 3) << 4);
                v[n][32 + p]      = (ql[32 + p] & 0xF) | (((h >> 2) & 3) << 4);
                v[n + 16][p]      = (ql[p] >> 4)       | (((h >> 4) & 3) << 4);
                v[n + 16][32 + p] = (ql[32 + p] >> 4)  | (((h >> 6) & 3) << 4);
            }
        }
        transpose_16x16_32bit(v);
        transpose_16x16_32bit(v + 16);
        // 1. lower 4 bits of 4 groups
        for (int n = 0; n < 32; n += 2) {
            for (int x = 0; x < 64; ++x) pb[x] = (v[n][x] & 0xF) | ((v[n + 1][x] & 0xF) << 4);
            pb += 64;
        }
        // 2. higher 2 bits of 4 groups
        for (int g = 0; g < 8; ++g) {
            for (int x = 0; x < 64; ++x) {
                ph[x] = ((v[g * 4 + 0][x] >> 4) & 3)
                      | (((v[g * 4 + 1][x] >> 4) & 3) << 2)
                      | (((v[g * 4 + 2][x] >> 4) & 3) << 4)
                      | (((v[g * 4 + 3][x] >> 4) & 3) << 6);
            }
            ph += 64;
        }
    }
    uint8_t * scales = out + (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N;
    for (int n = 0; n < TILE_N; ++n) {
        for (int k = 0; k < 16; ++k) scales[k * TILE_N + n] = static_cast<uint8_t>(b[n].scales[k]);
        put_u16(scales + 16 * TILE_N + 2 * n, b[n].d);
    }
}

// packed_B layout: quants {8, TILE_N, 16} uint8, scales {8, TILE_N} int8, d {TILE_N} half
void pack_B(uint8_t * out, const block_iq4_xs * b) {
    uint8_t v[16][64];
    uint8_t * pb = out;
    for (int k = 0; k < QK_K / 64; ++k) {
        for (int n = 0; n < TILE_N; ++n) {
            for (int p = 0; p < 16; ++p) {
                const uint8_t q0 = b[n].qs[k * 32 + p];
                const uint8_t q1 = b[n].qs[k * 32 + 16 + p];
                v[n][p]      = q0 & 0xF;
                v[n][16 + p] = q0 >> 4;
                v[n][32 + p] = q1 & 0xF;
                v[n][48 + p] = q1 >> 4;
            }
        }
        transpose_16x16_32bit(v);
        for (int n = 0; n < TILE_N; n += 2) {
            for (int x = 0; x < 64; ++x) pb[x] = v[n][x] | (v[n + 1][x] << 4);
            pb += 64;
        }
    }
    uint8_t * scales = out + (QK_K / 2) * TILE_N;
    for (int n = 0; n < TILE_N; ++n) {
        uint16_t sh = b[n].scales_h;
        for (int k = 0; k < 8; k += 2) {
            const int16_t ls1 = ((b[n].scales_l[k / 2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            const int16_t ls2 = ((b[n].scales_l[k / 2] >> 4) | ((sh << 2) & 0x30)) - 32;
            scales[(k + 0) * TILE_N + n] = static_cast<uint8_t>(static_cast<int8_t>(ls1));
            scales[(k + 1) * TILE_N + n] = static_cast<uint8_t>(static_cast<int8_t>(ls2));
            sh >>= 4;
        }
        put_u16(scales + 8 * TILE_N + 2 * n, b[n].d);
    }
}

// ---- driver -------------------------------------------------------------------------
struct QuantInfo {
    const char * name;
    int block_k;      // values per block
    int block_bytes;  // sizeof(block_*)
    int tile_bytes;   // get_tile_size<TB>()
    void (*convert)(uint8_t * out, const uint8_t * B, int64_t NB, int64_t KB);
};

// convert_B_packed_format: tiles over NB = N / TILE_N and KB = K / block_k
template <typename TB>
void convert_B_packed_format(uint8_t * out, const uint8_t * B, int64_t NB, int64_t KB, int tile_bytes) {
    TB blk[TILE_N];
    for (int64_t n = 0; n < NB; ++n) {
        for (int64_t k = 0; k < KB; ++k) {
            for (int r = 0; r < TILE_N; ++r) {
                std::memcpy(&blk[r], B + ((n * TILE_N + r) * KB + k) * sizeof(TB), sizeof(TB));
            }
            pack_B(out + (n * KB + k) * tile_bytes, blk);
        }
    }
}

#define TYLE_QUANT(NAME, TB, BLOCK_K, TILE_BYTES)                                              \
    {NAME, BLOCK_K, static_cast<int>(sizeof(TB)), TILE_BYTES,                                  \
     [](uint8_t * o, const uint8_t * b, int64_t nb, int64_t kb) {                              \
         convert_B_packed_format<TB>(o, b, nb, kb, TILE_BYTES);                                \
     }}

const QuantInfo kQuants[] = {
    TYLE_QUANT("q4_0",   block_q4_0,   32,   TILE_N * 18 + 0),
    TYLE_QUANT("q4_1",   block_q4_1,   32,   TILE_N * 20 + 0),
    TYLE_QUANT("q8_0",   block_q8_0,   32,   TILE_N * 34 + TILE_N * 4),
    TYLE_QUANT("q4_K",   block_q4_K,   QK_K, TILE_N * 144 + TILE_N * 4),
    TYLE_QUANT("q5_K",   block_q5_K,   QK_K, TILE_N * 176 + TILE_N * 4),
    TYLE_QUANT("q6_K",   block_q6_K,   QK_K, TILE_N * 210),
    TYLE_QUANT("iq4_xs", block_iq4_xs, QK_K, TILE_N * 136 + TILE_N * 2),
};

const QuantInfo * find_quant(const char * name) {
    for (const QuantInfo & q : kQuants) {
        if (std::strcmp(q.name, name) == 0) return &q;
    }
    PyErr_Format(PyExc_ValueError, "unsupported AMX source quant '%s'", name);
    return nullptr;
}

PyObject * py_quants(PyObject *, PyObject *) {
    const Py_ssize_t n = sizeof(kQuants) / sizeof(kQuants[0]);
    PyObject * t = PyTuple_New(n);
    if (!t) return nullptr;
    for (Py_ssize_t i = 0; i < n; ++i) PyTuple_SET_ITEM(t, i, PyUnicode_FromString(kQuants[i].name));
    return t;
}

PyObject * py_info(PyObject *, PyObject * args) {
    const char * name;
    if (!PyArg_ParseTuple(args, "s", &name)) return nullptr;
    const QuantInfo * q = find_quant(name);
    if (!q) return nullptr;
    return Py_BuildValue("(iii)", q->block_k, q->block_bytes, q->tile_bytes);
}

PyObject * py_pack(PyObject *, PyObject * args) {
    const char * name;
    Py_buffer buf;
    long long n, k;
    if (!PyArg_ParseTuple(args, "sy*LL", &name, &buf, &n, &k)) return nullptr;
    PyObject * result = nullptr;
    do {
        const QuantInfo * q = find_quant(name);
        if (!q) break;
        if (n <= 0 || k <= 0 || n % TILE_N != 0 || k % q->block_k != 0) {
            PyErr_Format(PyExc_ValueError, "(N, K)=(%lld, %lld) must be positive multiples of (%d, %d) for %s",
                         n, k, TILE_N, q->block_k, q->name);
            break;
        }
        const int64_t NB = n / TILE_N, KB = k / q->block_k;
        const int64_t expected = n * KB * q->block_bytes;
        if (buf.len != expected) {
            PyErr_Format(PyExc_ValueError, "%s blocks for (N, K)=(%lld, %lld) need %lld bytes, got %lld",
                         q->name, n, k, static_cast<long long>(expected), static_cast<long long>(buf.len));
            break;
        }
        const int64_t out_bytes = NB * KB * q->tile_bytes;
        result = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(out_bytes));
        if (!result) break;
        uint8_t * out = reinterpret_cast<uint8_t *>(PyBytes_AS_STRING(result));
        std::memset(out, 0, static_cast<size_t>(out_bytes));
        const uint8_t * in = static_cast<const uint8_t *>(buf.buf);
        Py_BEGIN_ALLOW_THREADS
        q->convert(out, in, NB, KB);
        Py_END_ALLOW_THREADS
    } while (false);
    PyBuffer_Release(&buf);
    return result;
}

PyMethodDef kMethods[] = {
    {"quants", py_quants, METH_NOARGS, "Supported AMX source quant names."},
    {"info", py_info, METH_VARARGS, "info(quant) -> (block_k, block_bytes, tile_bytes)"},
    {"pack", py_pack, METH_VARARGS, "pack(quant, blocks, n, k) -> bytes of the (N/16, K/block_k, tile_bytes) grid"},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kModule = {PyModuleDef_HEAD_INIT, "_amxpack", "TYLE AMX weight packing (llama.cpp mmq.cpp port)", -1, kMethods};

}  // namespace

PyMODINIT_FUNC PyInit__amxpack(void) { return PyModule_Create(&kModule); }
