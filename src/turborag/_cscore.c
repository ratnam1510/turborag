/*
 * TurboRAG fast scoring kernel — C extension for Python.
 *
 * LUT-based compressed dot product scoring on packed uint8 vectors.
 *
 * TWO ACCELERATION STRATEGIES:
 *
 * 1. Fused byte-level LUT (score_fused_*):
 *    Instead of extracting individual nibbles/bits and doing multiple LUT
 *    lookups per byte, we precompute a 256-entry table per byte position
 *    that maps each possible byte value directly to the total contribution
 *    from all dimensions packed in that byte.
 *    - 4-bit: 1 lookup per byte instead of 2 (2× fewer lookups)
 *    - 2-bit: 1 lookup per byte instead of 4 (4× fewer lookups)
 *
 * 2. Row-major packed storage with fused LUT:
 *    The fused byte LUT approach works well with row-major storage because
 *    each vector's bytes are contiguous, enabling good cache behavior when
 *    processing vectors in chunks.
 *
 * Build:
 *   gcc -O3 -march=native -funroll-loops -shared -fPIC -o _cscore.dylib _cscore.c
 */

#include <stdint.h>
#include <string.h>
#include <math.h>

/* ===================================================================== */
/*  FUSED BYTE LUT — the fast path                                       */
/*  Pre-fuse the per-dimension LUT into per-byte-position tables.        */
/* ===================================================================== */

/*
 * Build a fused LUT for 4-bit encoding (2 dims per byte).
 * Output: fused[byte_pos * 256 + byte_value] = sum of contributions.
 */
void build_fused_4bit(
    const double *lut,          /* (dim × 16) row-major */
    double       *fused,        /* (n_bytes × 256) output */
    int           dim)
{
    const int num_levels = 16;
    const int n_bytes = (dim + 1) / 2;

    for (int bi = 0; bi < n_bytes; bi++) {
        int d_even = bi * 2;
        int d_odd  = d_even + 1;
        const double *lut_even = lut + d_even * num_levels;
        const double *lut_odd  = (d_odd < dim) ? lut + d_odd * num_levels : 0;
        double *row = fused + bi * 256;

        for (int b = 0; b < 256; b++) {
            double val = lut_even[b & 0x0F];
            if (lut_odd) {
                val += lut_odd[(b >> 4) & 0x0F];
            }
            row[b] = val;
        }
    }
}

/*
 * Build a fused LUT for 2-bit encoding (4 dims per byte).
 */
void build_fused_2bit(
    const double *lut,          /* (dim × 4) row-major */
    double       *fused,        /* (n_bytes × 256) output */
    int           dim)
{
    const int num_levels = 4;
    const int n_bytes = (dim + 3) / 4;

    for (int bi = 0; bi < n_bytes; bi++) {
        int base_d = bi * 4;
        double *row = fused + bi * 256;

        for (int b = 0; b < 256; b++) {
            double val = 0.0;
            for (int sub = 0; sub < 4 && (base_d + sub) < dim; sub++) {
                int level = (b >> (sub * 2)) & 3;
                val += lut[(base_d + sub) * num_levels + level];
            }
            row[b] = val;
        }
    }
}

/*
 * Build a fused LUT for 3-bit encoding.
 * 3-bit values span byte boundaries, so we fuse per-dimension LUT entries
 * into groups of 8 dimensions (= 24 bits = 3 bytes exactly).
 * Each group reads 3 bytes (24 bits) and produces 8 dimension scores.
 * fused_3b[group * 16777216 + combined_24bit_value] would be huge, so
 * instead we split into a "triplet" LUT approach:
 *   For each triplet of bytes (3 bytes = 8 dims at 3 bits each),
 *   fused[triplet_idx * 256*256*256 + ...] — too big.
 *
 * For 3-bit, we fall back to a per-dimension approach with the fused
 * byte-boundary handling optimized.
 */

/* ===================================================================== */
/*  SCORING WITH FUSED LUT — one table lookup per packed byte            */
/* ===================================================================== */

/*
 * Score all vectors using a pre-built fused byte LUT.
 * Works for 2-bit and 4-bit where dims align to byte boundaries.
 *
 * Inner loop: for each byte position, sweep all vectors and
 * accumulate fused[byte_pos][packed_byte_value] into scores.
 */
void score_fused(
    const uint8_t *packed_db,   /* (n_vectors × bytes_per_vec), row-major */
    const double  *fused,       /* (n_bytes × 256) fused byte LUT */
    double        *scores_out,
    int            n_vectors,
    int            n_bytes,
    int            bytes_per_vec)
{
    /* Column-major: for each byte position, process all vectors */
    for (int bi = 0; bi < n_bytes; bi++) {
        const double *frow = fused + bi * 256;

        /* Process 4 vectors per iteration for ILP */
        int v = 0;
        int n4 = n_vectors & ~3;
        for (; v < n4; v += 4) {
            scores_out[v  ] += frow[packed_db[(size_t)(v  ) * bytes_per_vec + bi]];
            scores_out[v+1] += frow[packed_db[(size_t)(v+1) * bytes_per_vec + bi]];
            scores_out[v+2] += frow[packed_db[(size_t)(v+2) * bytes_per_vec + bi]];
            scores_out[v+3] += frow[packed_db[(size_t)(v+3) * bytes_per_vec + bi]];
        }
        for (; v < n_vectors; v++) {
            scores_out[v] += frow[packed_db[(size_t)v * bytes_per_vec + bi]];
        }
    }
}

/* ===================================================================== */
/*  3-bit scoring — column-major with byte-boundary spanning             */
/*  (no fused LUT; 3-bit doesn't align to byte boundaries)              */
/* ===================================================================== */

void score_lut_3bit(
    const uint8_t *packed_db,
    const double  *lut,
    double        *scores_out,
    int            n_vectors,
    int            dim,
    int            bytes_per_vec)
{
    const int num_levels = 8;
    const int mask = 0x07;
    int bit_offset = 0;

    for (int d = 0; d < dim; d++) {
        int byte_idx = bit_offset >> 3;
        int shift    = bit_offset & 7;
        int spills   = (shift + 3 > 8) ? 1 : 0;
        const double *lut_row = lut + d * num_levels;

        if (!spills) {
            int v = 0;
            int n4 = n_vectors & ~3;
            for (; v < n4; v += 4) {
                scores_out[v  ] += lut_row[(packed_db[(size_t)(v  )*bytes_per_vec + byte_idx] >> shift) & mask];
                scores_out[v+1] += lut_row[(packed_db[(size_t)(v+1)*bytes_per_vec + byte_idx] >> shift) & mask];
                scores_out[v+2] += lut_row[(packed_db[(size_t)(v+2)*bytes_per_vec + byte_idx] >> shift) & mask];
                scores_out[v+3] += lut_row[(packed_db[(size_t)(v+3)*bytes_per_vec + byte_idx] >> shift) & mask];
            }
            for (; v < n_vectors; v++) {
                scores_out[v] += lut_row[(packed_db[(size_t)v*bytes_per_vec + byte_idx] >> shift) & mask];
            }
        } else {
            int rshift = 8 - shift;
            for (int v = 0; v < n_vectors; v++) {
                const uint8_t *row = packed_db + (size_t)v * bytes_per_vec;
                int chunk = ((int)row[byte_idx] >> shift) | ((int)row[byte_idx + 1] << rshift);
                scores_out[v] += lut_row[chunk & mask];
            }
        }

        bit_offset += 3;
    }
}

/* ===================================================================== */
/*  Dispatch (called from Python via ctypes)                             */
/* ===================================================================== */

int score_lut_dispatch(
    const uint8_t *packed_db,
    const double  *lut,
    double        *scores_out,
    int            n_vectors,
    int            dim,
    int            bits,
    int            bytes_per_vec)
{
    memset(scores_out, 0, (size_t)n_vectors * sizeof(double));

    if (bits == 4) {
        int n_bytes = (dim + 1) / 2;
        double fused[n_bytes * 256];
        build_fused_4bit(lut, fused, dim);
        score_fused(packed_db, fused, scores_out, n_vectors, n_bytes, bytes_per_vec);
        return 0;
    } else if (bits == 2) {
        int n_bytes = (dim + 3) / 4;
        double fused[n_bytes * 256];
        build_fused_2bit(lut, fused, dim);
        score_fused(packed_db, fused, scores_out, n_vectors, n_bytes, bytes_per_vec);
        return 0;
    } else if (bits == 3) {
        score_lut_3bit(packed_db, lut, scores_out, n_vectors, dim, bytes_per_vec);
        return 0;
    }

    return -1;
}

/* ===================================================================== */
/*  Fused dispatch with pre-built fused LUT (avoids rebuild per call)    */
/* ===================================================================== */

int score_fused_dispatch(
    const uint8_t *packed_db,
    const double  *fused_lut,
    double        *scores_out,
    int            n_vectors,
    int            n_bytes,
    int            bytes_per_vec)
{
    memset(scores_out, 0, (size_t)n_vectors * sizeof(double));
    score_fused(packed_db, fused_lut, scores_out, n_vectors, n_bytes, bytes_per_vec);
    return 0;
}
