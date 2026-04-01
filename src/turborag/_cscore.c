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
#include <stdlib.h>
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
/*  3-bit scoring — fused byte-triplet approach                          */
/*                                                                       */
/*  Every 8 dims at 3 bits = 24 bits = exactly 3 bytes.                  */
/*  For each triplet of bytes we precompute 3 × 256-entry tables:        */
/*    fused3[triplet * 3 * 256 + byte_within_triplet * 256 + byte_value] */
/*  Each table entry sums the LUT contributions for the dimensions       */
/*  whose bits fall within that byte.                                    */
/*                                                                       */
/*  This reduces the inner loop from 8 LUT lookups per group to 3,       */
/*  and uses the same cache-friendly column-sweep as 2/4-bit.            */
/* ===================================================================== */

/*
 * Build fused byte-triplet LUT for 3-bit encoding.
 *
 * For a group of 8 dimensions packed into 3 bytes:
 *   byte0: bits [0..7]  → dims 0(bits 0-2), 1(bits 3-5), 2(bits 6-7 low)
 *   byte1: bits [8..15] → dim 2(bit 8 high), 3(bits 9-11), 4(bits 12-14), 5(bit 15 low)
 *   byte2: bits [16..23]→ dim 5(bits 16-17 high), 6(bits 18-20), 7(bits 21-23)
 *
 * fused_out: (n_triplets × 3 × 256) array.
 */
void build_fused_3bit(
    const double *lut,          /* (dim × 8) row-major */
    double       *fused_out,    /* (n_triplets × 3 × 256) output */
    int           dim)
{
    const int num_levels = 8;
    const int n_groups = (dim + 7) / 8;

    for (int g = 0; g < n_groups; g++) {
        int base_d = g * 8;
        double *fb0 = fused_out + (size_t)g * 3 * 256;
        double *fb1 = fb0 + 256;
        double *fb2 = fb1 + 256;

        /* Precompute which dims are extractable from each byte */
        for (int b = 0; b < 256; b++) {
            double v0 = 0.0, v1 = 0.0, v2 = 0.0;

            /* byte0: bit offsets 0,3,6 within the 24-bit group */
            /* dim+0 at bits 0-2 */
            if (base_d + 0 < dim)
                v0 += lut[(base_d + 0) * num_levels + (b & 0x07)];
            /* dim+1 at bits 3-5 */
            if (base_d + 1 < dim)
                v0 += lut[(base_d + 1) * num_levels + ((b >> 3) & 0x07)];
            /* dim+2 at bits 6-7 (low 2 bits of 3) */
            /* Need to combine with byte1 bit 0, can't fully resolve here.
               Store partial: extract the 2 low bits shifted into position. */

            /* byte1: */
            /* dim+2 spans byte0[6:7] and byte1[0] — handled below */
            /* dim+3 at bits 9-11 → byte1 bits 1-3 */
            if (base_d + 3 < dim)
                v1 += lut[(base_d + 3) * num_levels + ((b >> 1) & 0x07)];
            /* dim+4 at bits 12-14 → byte1 bits 4-6 */
            if (base_d + 4 < dim)
                v1 += lut[(base_d + 4) * num_levels + ((b >> 4) & 0x07)];
            /* dim+5 spans byte1[7] and byte2[0:1] — handled below */

            /* byte2: */
            /* dim+6 at bits 18-20 → byte2 bits 2-4 */
            if (base_d + 6 < dim)
                v2 += lut[(base_d + 6) * num_levels + ((b >> 2) & 0x07)];
            /* dim+7 at bits 21-23 → byte2 bits 5-7 */
            if (base_d + 7 < dim)
                v2 += lut[(base_d + 7) * num_levels + ((b >> 5) & 0x07)];

            fb0[b] = v0;
            fb1[b] = v1;
            fb2[b] = v2;
        }
    }
}

void score_lut_3bit(
    const uint8_t *packed_db,
    const double  *lut,
    double        *scores_out,
    int            n_vectors,
    int            dim,
    int            bytes_per_vec)
{
    const int num_levels = 8;
    const int n_groups = (dim + 7) / 8;

    /* Allocate fused triplet LUT + spanning-dim LUTs */
    double *fused3 = (double *)malloc((size_t)n_groups * 3 * 256 * sizeof(double));
    if (!fused3) {
        /* Fallback: simple per-dim path */
        const int mask = 0x07;
        int bit_offset = 0;
        for (int d = 0; d < dim; d++) {
            int byte_idx = bit_offset >> 3;
            int shift    = bit_offset & 7;
            const double *lut_row = lut + d * num_levels;
            if (shift + 3 <= 8) {
                for (int v = 0; v < n_vectors; v++)
                    scores_out[v] += lut_row[(packed_db[(size_t)v*bytes_per_vec + byte_idx] >> shift) & mask];
            } else {
                int rshift = 8 - shift;
                for (int v = 0; v < n_vectors; v++) {
                    const uint8_t *row = packed_db + (size_t)v * bytes_per_vec;
                    scores_out[v] += lut_row[((row[byte_idx] >> shift) | (row[byte_idx+1] << rshift)) & mask];
                }
            }
            bit_offset += 3;
        }
        return;
    }

    build_fused_3bit(lut, fused3, dim);

    /*
     * For each group of 8 dims, process 3 bytes with fused lookups
     * plus 2 spanning dimensions (dim+2 and dim+5) that cross byte
     * boundaries. The spanning dims are handled with direct extraction.
     */
    for (int g = 0; g < n_groups; g++) {
        int base_d = g * 8;
        int base_byte = g * 3;
        const double *fb0 = fused3 + (size_t)g * 3 * 256;
        const double *fb1 = fb0 + 256;
        const double *fb2 = fb1 + 256;

        /* Spanning dim pointers */
        const double *lut_d2 = (base_d + 2 < dim) ? lut + (base_d + 2) * num_levels : NULL;
        const double *lut_d5 = (base_d + 5 < dim) ? lut + (base_d + 5) * num_levels : NULL;

        int v = 0;
        int n4 = n_vectors & ~3;
        for (; v < n4; v += 4) {
            for (int vi = 0; vi < 4; vi++) {
                const uint8_t *row = packed_db + (size_t)(v + vi) * bytes_per_vec + base_byte;
                uint8_t b0 = row[0];
                uint8_t b1 = row[1];
                uint8_t b2 = row[2];
                double s = fb0[b0] + fb1[b1] + fb2[b2];
                /* dim+2: bits 6-8 → byte0[6:7] | byte1[0] */
                if (lut_d2) s += lut_d2[((b0 >> 6) | (b1 << 2)) & 0x07];
                /* dim+5: bits 15-17 → byte1[7] | byte2[0:1] */
                if (lut_d5) s += lut_d5[((b1 >> 7) | (b2 << 1)) & 0x07];
                scores_out[v + vi] += s;
            }
        }
        for (; v < n_vectors; v++) {
            const uint8_t *row = packed_db + (size_t)v * bytes_per_vec + base_byte;
            uint8_t b0 = row[0];
            uint8_t b1 = row[1];
            uint8_t b2 = row[2];
            double s = fb0[b0] + fb1[b1] + fb2[b2];
            if (lut_d2) s += lut_d2[((b0 >> 6) | (b1 << 2)) & 0x07];
            if (lut_d5) s += lut_d5[((b1 >> 7) | (b2 << 1)) & 0x07];
            scores_out[v] += s;
        }
    }

    free(fused3);
}

/* ===================================================================== */
/*  Binary Sketch Scorer — XOR + POPCNT Hamming distance                 */
/*  Used as a cheap pre-filter: scan all vectors, return Hamming          */
/*  distances, then refine only the top candidates with full LUT.        */
/* ===================================================================== */

/*
 * Compute Hamming distance between a query sketch and all database sketches.
 * sketch_db: (n_vectors × sketch_bytes), row-major packed sign bits.
 * sketch_q:  (sketch_bytes) query sketch.
 * distances_out: (n_vectors) output, lower = more similar.
 *
 * Uses __builtin_popcountll for fast bit counting.
 */
void hamming_scan(
    const uint8_t *sketch_db,
    const uint8_t *sketch_q,
    int32_t       *distances_out,
    int            n_vectors,
    int            sketch_bytes)
{
    /* Process in 8-byte (uint64) chunks for POPCNT efficiency */
    int n_words = sketch_bytes / 8;
    int tail_bytes = sketch_bytes % 8;

    for (int v = 0; v < n_vectors; v++) {
        const uint8_t *row = sketch_db + (size_t)v * sketch_bytes;
        int32_t dist = 0;

        /* Main loop: 8 bytes at a time */
        for (int w = 0; w < n_words; w++) {
            uint64_t db_word, q_word;
            memcpy(&db_word, row + w * 8, 8);
            memcpy(&q_word, sketch_q + w * 8, 8);
            dist += __builtin_popcountll(db_word ^ q_word);
        }

        /* Tail bytes */
        for (int b = n_words * 8; b < sketch_bytes; b++) {
            dist += __builtin_popcount(row[b] ^ sketch_q[b]);
        }

        distances_out[v] = dist;
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
        double *fused = (double *)malloc((size_t)n_bytes * 256 * sizeof(double));
        if (!fused) return -2;
        build_fused_4bit(lut, fused, dim);
        score_fused(packed_db, fused, scores_out, n_vectors, n_bytes, bytes_per_vec);
        free(fused);
        return 0;
    } else if (bits == 2) {
        int n_bytes = (dim + 3) / 4;
        double *fused = (double *)malloc((size_t)n_bytes * 256 * sizeof(double));
        if (!fused) return -2;
        build_fused_2bit(lut, fused, dim);
        score_fused(packed_db, fused, scores_out, n_vectors, n_bytes, bytes_per_vec);
        free(fused);
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

/* ===================================================================== */
/*  3-bit scoring with PREBUILT fused table                              */
/*                                                                       */
/*  Identical logic to score_lut_3bit but accepts an externally built    */
/*  fused 3-bit table so callers can reuse it across shard scans.        */
/*  The original LUT is still needed for the two spanning dimensions     */
/*  (dim+2 and dim+5 per group) that cross byte boundaries.             */
/* ===================================================================== */

int score_fused_3bit_dispatch(
    const uint8_t *packed_db,
    const double  *fused3,       /* (n_groups × 3 × 256) prebuilt fused table */
    const double  *lut,          /* (dim × 8) original LUT for spanning dims */
    double        *scores_out,
    int            n_vectors,
    int            dim,
    int            bytes_per_vec)
{
    const int num_levels = 8;
    const int n_groups = (dim + 7) / 8;

    memset(scores_out, 0, (size_t)n_vectors * sizeof(double));

    for (int g = 0; g < n_groups; g++) {
        int base_d = g * 8;
        int base_byte = g * 3;
        const double *fb0 = fused3 + (size_t)g * 3 * 256;
        const double *fb1 = fb0 + 256;
        const double *fb2 = fb1 + 256;

        /* Spanning dim pointers */
        const double *lut_d2 = (base_d + 2 < dim) ? lut + (base_d + 2) * num_levels : NULL;
        const double *lut_d5 = (base_d + 5 < dim) ? lut + (base_d + 5) * num_levels : NULL;

        int v = 0;
        int n4 = n_vectors & ~3;
        for (; v < n4; v += 4) {
            for (int vi = 0; vi < 4; vi++) {
                const uint8_t *row = packed_db + (size_t)(v + vi) * bytes_per_vec + base_byte;
                uint8_t b0 = row[0];
                uint8_t b1 = row[1];
                uint8_t b2 = row[2];
                double s = fb0[b0] + fb1[b1] + fb2[b2];
                /* dim+2: bits 6-8 → byte0[6:7] | byte1[0] */
                if (lut_d2) s += lut_d2[((b0 >> 6) | (b1 << 2)) & 0x07];
                /* dim+5: bits 15-17 → byte1[7] | byte2[0:1] */
                if (lut_d5) s += lut_d5[((b1 >> 7) | (b2 << 1)) & 0x07];
                scores_out[v + vi] += s;
            }
        }
        for (; v < n_vectors; v++) {
            const uint8_t *row = packed_db + (size_t)v * bytes_per_vec + base_byte;
            uint8_t b0 = row[0];
            uint8_t b1 = row[1];
            uint8_t b2 = row[2];
            double s = fb0[b0] + fb1[b1] + fb2[b2];
            if (lut_d2) s += lut_d2[((b0 >> 6) | (b1 << 2)) & 0x07];
            if (lut_d5) s += lut_d5[((b1 >> 7) | (b2 << 1)) & 0x07];
            scores_out[v] += s;
        }
    }

    return 0;
}
