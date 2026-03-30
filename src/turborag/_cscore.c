/*
 * TurboRAG fast scoring kernel — C extension for Python.
 *
 * LUT-based compressed dot product scoring operating directly on packed
 * uint8 database vectors.
 *
 * KEY DESIGN: column-major traversal (for d → for v) so that the
 * scores[] array stays in L1/L2 cache. Each dimension reads one LUT row
 * (tiny — 8-16 doubles) and sweeps through all vectors. The packed byte
 * column is also sequential in memory when row-major packed, since every
 * vector uses the same byte index for the same dimension.
 *
 * This layout matches what NumPy BLAS does: it processes one "column"
 * (one query dimension multiplied against all DB rows) at a time, which
 * keeps the accumulator array hot.
 *
 * Build:
 *   gcc -O3 -march=native -funroll-loops -shared -fPIC -o _cscore.dylib _cscore.c
 */

#include <stdint.h>
#include <string.h>

/* --------------------------------------------------------------------- */
/*  4-bit scoring — column-major: for dim → for vectors                  */
/* --------------------------------------------------------------------- */
void score_lut_4bit(
    const uint8_t *packed_db,
    const double  *lut,            /* (dim × 16) row-major */
    double        *scores_out,
    int            n_vectors,
    int            dim,
    int            bytes_per_vec)
{
    const int num_levels = 16;
    const int full_bytes = dim >> 1;
    const int has_tail   = dim & 1;

    /* Process two dimensions per byte (low nibble = even dim, high = odd) */
    for (int byte_idx = 0; byte_idx < full_bytes; byte_idx++) {
        int d_even = byte_idx << 1;
        int d_odd  = d_even + 1;

        const double *lut_even = lut + d_even * num_levels;
        const double *lut_odd  = (d_odd < dim) ? lut + d_odd * num_levels : 0;

        for (int v = 0; v < n_vectors; v++) {
            uint8_t b = packed_db[(size_t)v * bytes_per_vec + byte_idx];
            scores_out[v] += lut_even[b & 0x0F];
            if (lut_odd) {
                scores_out[v] += lut_odd[(b >> 4) & 0x0F];
            }
        }
    }

    /* Odd trailing dimension */
    if (has_tail) {
        int byte_idx = full_bytes;
        int d = dim - 1;
        const double *lut_row = lut + d * num_levels;
        for (int v = 0; v < n_vectors; v++) {
            uint8_t b = packed_db[(size_t)v * bytes_per_vec + byte_idx];
            scores_out[v] += lut_row[b & 0x0F];
        }
    }
}

/* --------------------------------------------------------------------- */
/*  2-bit scoring — column-major                                         */
/* --------------------------------------------------------------------- */
void score_lut_2bit(
    const uint8_t *packed_db,
    const double  *lut,            /* (dim × 4) row-major */
    double        *scores_out,
    int            n_vectors,
    int            dim,
    int            bytes_per_vec)
{
    const int num_levels = 4;
    const int full_bytes = dim >> 2;
    const int tail_dims  = dim & 3;

    for (int byte_idx = 0; byte_idx < full_bytes; byte_idx++) {
        int base_d = byte_idx << 2;
        const double *lut0 = lut + (base_d    ) * num_levels;
        const double *lut1 = lut + (base_d + 1) * num_levels;
        const double *lut2 = lut + (base_d + 2) * num_levels;
        const double *lut3 = lut + (base_d + 3) * num_levels;

        for (int v = 0; v < n_vectors; v++) {
            uint8_t b = packed_db[(size_t)v * bytes_per_vec + byte_idx];
            scores_out[v] += lut0[ b       & 3];
            scores_out[v] += lut1[(b >> 2) & 3];
            scores_out[v] += lut2[(b >> 4) & 3];
            scores_out[v] += lut3[(b >> 6) & 3];
        }
    }

    /* Remaining 1-3 dimensions */
    if (tail_dims > 0) {
        int byte_idx = full_bytes;
        int base_d = full_bytes << 2;
        for (int t = 0; t < tail_dims; t++) {
            const double *lut_row = lut + (base_d + t) * num_levels;
            int shift = t << 1;
            for (int v = 0; v < n_vectors; v++) {
                uint8_t b = packed_db[(size_t)v * bytes_per_vec + byte_idx];
                scores_out[v] += lut_row[(b >> shift) & 3];
            }
        }
    }
}

/* --------------------------------------------------------------------- */
/*  3-bit scoring — column-major with byte-boundary spanning             */
/* --------------------------------------------------------------------- */
void score_lut_3bit(
    const uint8_t *packed_db,
    const double  *lut,            /* (dim × 8) row-major */
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
            /* Fast path: no byte spanning */
            for (int v = 0; v < n_vectors; v++) {
                uint8_t b = packed_db[(size_t)v * bytes_per_vec + byte_idx];
                scores_out[v] += lut_row[(b >> shift) & mask];
            }
        } else {
            /* Spans a byte boundary */
            for (int v = 0; v < n_vectors; v++) {
                const uint8_t *row = packed_db + (size_t)v * bytes_per_vec;
                int chunk = ((int)row[byte_idx]) >> shift;
                chunk |= ((int)row[byte_idx + 1]) << (8 - shift);
                scores_out[v] += lut_row[chunk & mask];
            }
        }

        bit_offset += 3;
    }
}

/* --------------------------------------------------------------------- */
/*  Dispatch                                                              */
/* --------------------------------------------------------------------- */
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

    switch (bits) {
        case 4:
            score_lut_4bit(packed_db, lut, scores_out, n_vectors, dim, bytes_per_vec);
            return 0;
        case 3:
            score_lut_3bit(packed_db, lut, scores_out, n_vectors, dim, bytes_per_vec);
            return 0;
        case 2:
            score_lut_2bit(packed_db, lut, scores_out, n_vectors, dim, bytes_per_vec);
            return 0;
        default:
            return -1;
    }
}
