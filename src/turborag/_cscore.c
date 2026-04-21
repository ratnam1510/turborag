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
#include <pthread.h>

static void sort_topk_desc(int32_t *indices_out, float *scores_out, int count)
{
    for (int i = 1; i < count; i++) {
        int32_t idx = indices_out[i];
        float score = scores_out[i];
        int j = i - 1;
        while (j >= 0 && scores_out[j] < score) {
            indices_out[j + 1] = indices_out[j];
            scores_out[j + 1] = scores_out[j];
            j--;
        }
        indices_out[j + 1] = idx;
        scores_out[j + 1] = score;
    }
}

static inline double score_row_fused_3bit_fullgroups(
    const uint8_t *row,
    const double  *fused3,
    const double  *lut,
    int            n_groups)
{
    double total = 0.0;
    const double *fused_group = fused3;
    const double *lut_group = lut;

    for (int g = 0; g < n_groups; g++) {
        uint8_t b0 = row[0];
        uint8_t b1 = row[1];
        uint8_t b2 = row[2];

        total += fused_group[b0];
        total += fused_group[256 + b1];
        total += fused_group[512 + b2];
        total += lut_group[16 + (((b0 >> 6) | (b1 << 2)) & 0x07)];
        total += lut_group[40 + (((b1 >> 7) | (b2 << 1)) & 0x07)];

        row += 3;
        fused_group += 3 * 256;
        lut_group += 8 * 8;
    }

    return total;
}

static inline float score_row_fused_3bit_fullgroups_f32(
    const uint8_t *row,
    const float   *fused3,
    const float   *lut,
    int            n_groups)
{
    float total = 0.0f;
    const float *fused_group = fused3;
    const float *lut_group = lut;

    for (int g = 0; g < n_groups; g++) {
        uint8_t b0 = row[0];
        uint8_t b1 = row[1];
        uint8_t b2 = row[2];

        total += fused_group[b0];
        total += fused_group[256 + b1];
        total += fused_group[512 + b2];
        total += lut_group[16 + (((b0 >> 6) | (b1 << 2)) & 0x07)];
        total += lut_group[40 + (((b1 >> 7) | (b2 << 1)) & 0x07)];

        row += 3;
        fused_group += 3 * 256;
        lut_group += 8 * 8;
    }

    return total;
}

static inline float score_row_fused_3bit_6bit_fullgroups_f32(
    const uint8_t *row,
    const float   *fused6,
    int            n_groups)
{
    float total = 0.0f;
    const float *group_table = fused6;

    for (int g = 0; g < n_groups; g++) {
        uint8_t b0 = row[0];
        uint8_t b1 = row[1];
        uint8_t b2 = row[2];
        uint8_t idx01 = b0 & 0x3F;
        uint8_t idx23 = ((b0 >> 6) | (b1 << 2)) & 0x3F;
        uint8_t idx45 = ((b1 >> 4) | (b2 << 4)) & 0x3F;
        uint8_t idx67 = (b2 >> 2) & 0x3F;

        total += group_table[idx01];
        total += group_table[64 + idx23];
        total += group_table[128 + idx45];
        total += group_table[192 + idx67];

        row += 3;
        group_table += 4 * 64;
    }

    return total;
}

typedef struct {
    const uint8_t *packed_db;
    const float   *fused3;
    const float   *lut;
    int32_t       *indices_out;
    float         *scores_out;
    int            start;
    int            end;
    int            dim;
    int            bytes_per_vec;
    int            k;
    int            found;
} topk_thread_ctx_f32_t;

typedef struct {
    const uint8_t *packed_db;
    const float   *fused6;
    int32_t       *indices_out;
    float         *scores_out;
    int            start;
    int            end;
    int            dim;
    int            bytes_per_vec;
    int            k;
    int            found;
} topk_thread_ctx_6bit_f32_t;

typedef struct {
    const uint8_t *packed_db;
    const double  *fused3;
    const double  *lut;
    int32_t       *indices_out;
    float         *scores_out;
    int            start;
    int            end;
    int            dim;
    int            bytes_per_vec;
    int            k;
    int            found;
} topk_thread_ctx_t;

static int score_fused_3bit_topk_range(
    const uint8_t *packed_db,
    const double  *fused3,
    const double  *lut,
    int32_t       *indices_out,
    float         *scores_out,
    int            start,
    int            end,
    int            dim,
    int            bytes_per_vec,
    int            k)
{
    const int num_levels = 8;
    const int n_groups = (dim + 7) / 8;
    const int full_groups = (dim & 7) == 0;
    int found = 0;
    float min_score = 0.0f;
    int min_pos = 0;

    for (int v = start; v < end; v++) {
        const uint8_t *row = packed_db + (size_t)v * bytes_per_vec;
        double total = full_groups ? score_row_fused_3bit_fullgroups(row, fused3, lut, n_groups) : 0.0;

        if (!full_groups) {
            for (int g = 0; g < n_groups; g++) {
                int base_d = g * 8;
                int base_byte = g * 3;
                const double *fb0 = fused3 + (size_t)g * 3 * 256;
                const double *fb1 = fb0 + 256;
                const double *fb2 = fb1 + 256;
                const double *lut_d2 = (base_d + 2 < dim) ? lut + (base_d + 2) * num_levels : NULL;
                const double *lut_d5 = (base_d + 5 < dim) ? lut + (base_d + 5) * num_levels : NULL;
                uint8_t b0 = row[base_byte];
                uint8_t b1 = row[base_byte + 1];
                uint8_t b2 = row[base_byte + 2];

                total += fb0[b0] + fb1[b1] + fb2[b2];
                if (lut_d2) total += lut_d2[((b0 >> 6) | (b1 << 2)) & 0x07];
                if (lut_d5) total += lut_d5[((b1 >> 7) | (b2 << 1)) & 0x07];
            }
        }

        float score = (float)total;
        if (found < k) {
            indices_out[found] = v;
            scores_out[found] = score;
            if (found == 0 || score < min_score) {
                min_score = score;
                min_pos = found;
            }
            found++;
            continue;
        }

        if (score <= min_score) {
            continue;
        }

        indices_out[min_pos] = v;
        scores_out[min_pos] = score;
        min_pos = 0;
        min_score = scores_out[0];
        for (int i = 1; i < k; i++) {
            if (scores_out[i] < min_score) {
                min_score = scores_out[i];
                min_pos = i;
            }
        }
    }

    sort_topk_desc(indices_out, scores_out, found);
    return found;
}

static int score_fused_3bit_topk_range_f32(
    const uint8_t *packed_db,
    const float   *fused3,
    const float   *lut,
    int32_t       *indices_out,
    float         *scores_out,
    int            start,
    int            end,
    int            dim,
    int            bytes_per_vec,
    int            k)
{
    const int num_levels = 8;
    const int n_groups = (dim + 7) / 8;
    const int full_groups = (dim & 7) == 0;
    int found = 0;
    float min_score = 0.0f;
    int min_pos = 0;

    for (int v = start; v < end; v++) {
        const uint8_t *row = packed_db + (size_t)v * bytes_per_vec;
        float total = full_groups ? score_row_fused_3bit_fullgroups_f32(row, fused3, lut, n_groups) : 0.0f;

        if (!full_groups) {
            for (int g = 0; g < n_groups; g++) {
                int base_d = g * 8;
                int base_byte = g * 3;
                const float *fb0 = fused3 + (size_t)g * 3 * 256;
                const float *fb1 = fb0 + 256;
                const float *fb2 = fb1 + 256;
                const float *lut_d2 = (base_d + 2 < dim) ? lut + (base_d + 2) * num_levels : NULL;
                const float *lut_d5 = (base_d + 5 < dim) ? lut + (base_d + 5) * num_levels : NULL;
                uint8_t b0 = row[base_byte];
                uint8_t b1 = row[base_byte + 1];
                uint8_t b2 = row[base_byte + 2];

                total += fb0[b0] + fb1[b1] + fb2[b2];
                if (lut_d2) total += lut_d2[((b0 >> 6) | (b1 << 2)) & 0x07];
                if (lut_d5) total += lut_d5[((b1 >> 7) | (b2 << 1)) & 0x07];
            }
        }

        if (found < k) {
            indices_out[found] = v;
            scores_out[found] = total;
            if (found == 0 || total < min_score) {
                min_score = total;
                min_pos = found;
            }
            found++;
            continue;
        }

        if (total <= min_score) {
            continue;
        }

        indices_out[min_pos] = v;
        scores_out[min_pos] = total;
        min_pos = 0;
        min_score = scores_out[0];
        for (int i = 1; i < k; i++) {
            if (scores_out[i] < min_score) {
                min_score = scores_out[i];
                min_pos = i;
            }
        }
    }

    sort_topk_desc(indices_out, scores_out, found);
    return found;
}

static int score_fused_3bit_topk_range_6bit_f32(
    const uint8_t *packed_db,
    const float   *fused6,
    int32_t       *indices_out,
    float         *scores_out,
    int            start,
    int            end,
    int            dim,
    int            bytes_per_vec,
    int            k)
{
    const int n_groups = dim / 8;
    int found = 0;
    float min_score = 0.0f;
    int min_pos = 0;

    for (int v = start; v < end; v++) {
        const uint8_t *row = packed_db + (size_t)v * bytes_per_vec;
        float score = score_row_fused_3bit_6bit_fullgroups_f32(row, fused6, n_groups);

        if (found < k) {
            indices_out[found] = v;
            scores_out[found] = score;
            if (found == 0 || score < min_score) {
                min_score = score;
                min_pos = found;
            }
            found++;
            continue;
        }

        if (score <= min_score) {
            continue;
        }

        indices_out[min_pos] = v;
        scores_out[min_pos] = score;
        min_pos = 0;
        min_score = scores_out[0];
        for (int i = 1; i < k; i++) {
            if (scores_out[i] < min_score) {
                min_score = scores_out[i];
                min_pos = i;
            }
        }
    }

    sort_topk_desc(indices_out, scores_out, found);
    return found;
}

static void *score_fused_3bit_topk_worker(void *arg)
{
    topk_thread_ctx_t *ctx = (topk_thread_ctx_t *)arg;
    ctx->found = score_fused_3bit_topk_range(
        ctx->packed_db,
        ctx->fused3,
        ctx->lut,
        ctx->indices_out,
        ctx->scores_out,
        ctx->start,
        ctx->end,
        ctx->dim,
        ctx->bytes_per_vec,
        ctx->k
    );
    return NULL;
}

static void *score_fused_3bit_topk_worker_f32(void *arg)
{
    topk_thread_ctx_f32_t *ctx = (topk_thread_ctx_f32_t *)arg;
    ctx->found = score_fused_3bit_topk_range_f32(
        ctx->packed_db,
        ctx->fused3,
        ctx->lut,
        ctx->indices_out,
        ctx->scores_out,
        ctx->start,
        ctx->end,
        ctx->dim,
        ctx->bytes_per_vec,
        ctx->k
    );
    return NULL;
}

static void *score_fused_3bit_topk_worker_6bit_f32(void *arg)
{
    topk_thread_ctx_6bit_f32_t *ctx = (topk_thread_ctx_6bit_f32_t *)arg;
    ctx->found = score_fused_3bit_topk_range_6bit_f32(
        ctx->packed_db,
        ctx->fused6,
        ctx->indices_out,
        ctx->scores_out,
        ctx->start,
        ctx->end,
        ctx->dim,
        ctx->bytes_per_vec,
        ctx->k
    );
    return NULL;
}

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

void build_fused_3bit_f32(
    const float *lut,         /* (dim × 8) row-major */
    float       *fused_out,   /* (n_triplets × 3 × 256) output */
    int          dim)
{
    const int num_levels = 8;
    const int n_groups = (dim + 7) / 8;

    for (int g = 0; g < n_groups; g++) {
        int base_d = g * 8;
        float *fb0 = fused_out + (size_t)g * 3 * 256;
        float *fb1 = fb0 + 256;
        float *fb2 = fb1 + 256;

        for (int b = 0; b < 256; b++) {
            float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f;

            if (base_d + 0 < dim)
                v0 += lut[(base_d + 0) * num_levels + (b & 0x07)];
            if (base_d + 1 < dim)
                v0 += lut[(base_d + 1) * num_levels + ((b >> 3) & 0x07)];

            if (base_d + 3 < dim)
                v1 += lut[(base_d + 3) * num_levels + ((b >> 1) & 0x07)];
            if (base_d + 4 < dim)
                v1 += lut[(base_d + 4) * num_levels + ((b >> 4) & 0x07)];

            if (base_d + 6 < dim)
                v2 += lut[(base_d + 6) * num_levels + ((b >> 2) & 0x07)];
            if (base_d + 7 < dim)
                v2 += lut[(base_d + 7) * num_levels + ((b >> 5) & 0x07)];

            fb0[b] = v0;
            fb1[b] = v1;
            fb2[b] = v2;
        }
    }
}

void build_fused_3bit_6bit_f32(
    const float *lut,        /* (dim × 8) row-major */
    float       *fused_out,  /* (n_groups × 4 × 64) output */
    int          dim)
{
    const int n_groups = dim / 8;

    for (int g = 0; g < n_groups; g++) {
        const float *lut_group = lut + (size_t)g * 8 * 8;
        float *pair01 = fused_out + (size_t)g * 4 * 64;
        float *pair23 = pair01 + 64;
        float *pair45 = pair23 + 64;
        float *pair67 = pair45 + 64;

        for (int code = 0; code < 64; code++) {
            pair01[code] =
                lut_group[((code >> 0) & 0x07)] +
                lut_group[8 + ((code >> 3) & 0x07)];
            pair23[code] =
                lut_group[16 + ((code >> 0) & 0x07)] +
                lut_group[24 + ((code >> 3) & 0x07)];
            pair45[code] =
                lut_group[32 + ((code >> 0) & 0x07)] +
                lut_group[40 + ((code >> 3) & 0x07)];
            pair67[code] =
                lut_group[48 + ((code >> 0) & 0x07)] +
                lut_group[56 + ((code >> 3) & 0x07)];
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
    const int full_groups = (dim & 7) == 0;

    memset(scores_out, 0, (size_t)n_vectors * sizeof(double));

    if (full_groups) {
        for (int v = 0; v < n_vectors; v++) {
            const uint8_t *row = packed_db + (size_t)v * bytes_per_vec;
            scores_out[v] = score_row_fused_3bit_fullgroups(row, fused3, lut, n_groups);
        }
        return 0;
    }

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

int score_fused_3bit_topk_dispatch(
    const uint8_t *packed_db,
    const double  *fused3,       /* (n_groups × 3 × 256) prebuilt fused table */
    const double  *lut,          /* (dim × 8) original LUT for spanning dims */
    int32_t       *indices_out,
    float         *scores_out,
    int            n_vectors,
    int            dim,
    int            bytes_per_vec,
    int            k,
    int            num_threads)
{
    if (k <= 0) {
        return 0;
    }

    if (num_threads <= 1 || n_vectors < 32768) {
        return score_fused_3bit_topk_range(
            packed_db, fused3, lut, indices_out, scores_out, 0, n_vectors, dim, bytes_per_vec, k
        );
    }

    if (num_threads > n_vectors) {
        num_threads = n_vectors;
    }

    pthread_t *threads = (pthread_t *)malloc((size_t)num_threads * sizeof(pthread_t));
    int *thread_created = (int *)calloc((size_t)num_threads, sizeof(int));
    topk_thread_ctx_t *ctxs = (topk_thread_ctx_t *)calloc((size_t)num_threads, sizeof(topk_thread_ctx_t));
    int32_t *local_indices = (int32_t *)malloc((size_t)num_threads * (size_t)k * sizeof(int32_t));
    float *local_scores = (float *)malloc((size_t)num_threads * (size_t)k * sizeof(float));
    if (!threads || !thread_created || !ctxs || !local_indices || !local_scores) {
        free(threads);
        free(thread_created);
        free(ctxs);
        free(local_indices);
        free(local_scores);
        return score_fused_3bit_topk_range(
            packed_db, fused3, lut, indices_out, scores_out, 0, n_vectors, dim, bytes_per_vec, k
        );
    }

    int chunk = (n_vectors + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; t++) {
        int start = t * chunk;
        int end = start + chunk;
        if (start >= n_vectors) {
            break;
        }
        if (end > n_vectors) {
            end = n_vectors;
        }
        ctxs[t].packed_db = packed_db;
        ctxs[t].fused3 = fused3;
        ctxs[t].lut = lut;
        ctxs[t].indices_out = local_indices + (size_t)t * k;
        ctxs[t].scores_out = local_scores + (size_t)t * k;
        ctxs[t].start = start;
        ctxs[t].end = end;
        ctxs[t].dim = dim;
        ctxs[t].bytes_per_vec = bytes_per_vec;
        ctxs[t].k = k;
        ctxs[t].found = 0;
        if (pthread_create(&threads[t], NULL, score_fused_3bit_topk_worker, &ctxs[t]) != 0) {
            ctxs[t].found = score_fused_3bit_topk_range(
                packed_db, fused3, lut, ctxs[t].indices_out, ctxs[t].scores_out, start, end, dim, bytes_per_vec, k
            );
        } else {
            thread_created[t] = 1;
        }
    }

    for (int t = 0; t < num_threads; t++) {
        if (thread_created[t]) {
            pthread_join(threads[t], NULL);
        }
    }

    int found = 0;
    float min_score = 0.0f;
    int min_pos = 0;
    for (int t = 0; t < num_threads; t++) {
        int local_found = ctxs[t].found;
        int32_t *idxs = local_indices + (size_t)t * k;
        float *scores = local_scores + (size_t)t * k;
        for (int i = 0; i < local_found; i++) {
            float score = scores[i];
            if (found < k) {
                indices_out[found] = idxs[i];
                scores_out[found] = score;
                if (found == 0 || score < min_score) {
                    min_score = score;
                    min_pos = found;
                }
                found++;
                continue;
            }
            if (score <= min_score) {
                continue;
            }
            indices_out[min_pos] = idxs[i];
            scores_out[min_pos] = score;
            min_pos = 0;
            min_score = scores_out[0];
            for (int j = 1; j < k; j++) {
                if (scores_out[j] < min_score) {
                    min_score = scores_out[j];
                    min_pos = j;
                }
            }
        }
    }

    free(threads);
    free(thread_created);
    free(ctxs);
    free(local_indices);
    free(local_scores);

    sort_topk_desc(indices_out, scores_out, found);
    return found;
}

int score_fused_3bit_topk_dispatch_f32(
    const uint8_t *packed_db,
    const float   *fused3,
    const float   *lut,
    int32_t       *indices_out,
    float         *scores_out,
    int            n_vectors,
    int            dim,
    int            bytes_per_vec,
    int            k,
    int            num_threads)
{
    if (k <= 0) {
        return 0;
    }
    if (num_threads <= 1 || n_vectors < 32768) {
        return score_fused_3bit_topk_range_f32(
            packed_db, fused3, lut, indices_out, scores_out, 0, n_vectors, dim, bytes_per_vec, k
        );
    }

    if (num_threads > n_vectors) {
        num_threads = n_vectors;
    }

    pthread_t *threads = (pthread_t *)malloc((size_t)num_threads * sizeof(pthread_t));
    int *thread_created = (int *)calloc((size_t)num_threads, sizeof(int));
    topk_thread_ctx_f32_t *ctxs = (topk_thread_ctx_f32_t *)calloc((size_t)num_threads, sizeof(topk_thread_ctx_f32_t));
    int32_t *local_indices = (int32_t *)malloc((size_t)num_threads * (size_t)k * sizeof(int32_t));
    float *local_scores = (float *)malloc((size_t)num_threads * (size_t)k * sizeof(float));
    if (!threads || !thread_created || !ctxs || !local_indices || !local_scores) {
        free(threads);
        free(thread_created);
        free(ctxs);
        free(local_indices);
        free(local_scores);
        return score_fused_3bit_topk_range_f32(
            packed_db, fused3, lut, indices_out, scores_out, 0, n_vectors, dim, bytes_per_vec, k
        );
    }

    int chunk = (n_vectors + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; t++) {
        int start = t * chunk;
        int end = start + chunk;
        if (start >= n_vectors) {
            break;
        }
        if (end > n_vectors) {
            end = n_vectors;
        }
        ctxs[t].packed_db = packed_db;
        ctxs[t].fused3 = fused3;
        ctxs[t].lut = lut;
        ctxs[t].indices_out = local_indices + (size_t)t * k;
        ctxs[t].scores_out = local_scores + (size_t)t * k;
        ctxs[t].start = start;
        ctxs[t].end = end;
        ctxs[t].dim = dim;
        ctxs[t].bytes_per_vec = bytes_per_vec;
        ctxs[t].k = k;
        ctxs[t].found = 0;
        if (pthread_create(&threads[t], NULL, score_fused_3bit_topk_worker_f32, &ctxs[t]) != 0) {
            ctxs[t].found = score_fused_3bit_topk_range_f32(
                packed_db, fused3, lut, ctxs[t].indices_out, ctxs[t].scores_out, start, end, dim, bytes_per_vec, k
            );
        } else {
            thread_created[t] = 1;
        }
    }

    for (int t = 0; t < num_threads; t++) {
        if (thread_created[t]) {
            pthread_join(threads[t], NULL);
        }
    }

    int found = 0;
    float min_score = 0.0f;
    int min_pos = 0;
    for (int t = 0; t < num_threads; t++) {
        int local_found = ctxs[t].found;
        int32_t *idxs = local_indices + (size_t)t * k;
        float *scores = local_scores + (size_t)t * k;
        for (int i = 0; i < local_found; i++) {
            float score = scores[i];
            if (found < k) {
                indices_out[found] = idxs[i];
                scores_out[found] = score;
                if (found == 0 || score < min_score) {
                    min_score = score;
                    min_pos = found;
                }
                found++;
                continue;
            }
            if (score <= min_score) {
                continue;
            }
            indices_out[min_pos] = idxs[i];
            scores_out[min_pos] = score;
            min_pos = 0;
            min_score = scores_out[0];
            for (int j = 1; j < k; j++) {
                if (scores_out[j] < min_score) {
                    min_score = scores_out[j];
                    min_pos = j;
                }
            }
        }
    }

    free(threads);
    free(thread_created);
    free(ctxs);
    free(local_indices);
    free(local_scores);

    sort_topk_desc(indices_out, scores_out, found);
    return found;
}

int score_fused_3bit_topk_dispatch_6bit_f32(
    const uint8_t *packed_db,
    const float   *fused6,
    int32_t       *indices_out,
    float         *scores_out,
    int            n_vectors,
    int            dim,
    int            bytes_per_vec,
    int            k,
    int            num_threads)
{
    if (k <= 0) {
        return 0;
    }
    if ((dim & 7) != 0) {
        return -1;
    }

    if (num_threads <= 1 || n_vectors < 32768) {
        return score_fused_3bit_topk_range_6bit_f32(
            packed_db, fused6, indices_out, scores_out, 0, n_vectors, dim, bytes_per_vec, k
        );
    }

    if (num_threads > n_vectors) {
        num_threads = n_vectors;
    }

    pthread_t *threads = (pthread_t *)malloc((size_t)num_threads * sizeof(pthread_t));
    int *thread_created = (int *)calloc((size_t)num_threads, sizeof(int));
    topk_thread_ctx_6bit_f32_t *ctxs = (topk_thread_ctx_6bit_f32_t *)calloc((size_t)num_threads, sizeof(topk_thread_ctx_6bit_f32_t));
    int32_t *local_indices = (int32_t *)malloc((size_t)num_threads * (size_t)k * sizeof(int32_t));
    float *local_scores = (float *)malloc((size_t)num_threads * (size_t)k * sizeof(float));
    if (!threads || !thread_created || !ctxs || !local_indices || !local_scores) {
        free(threads);
        free(thread_created);
        free(ctxs);
        free(local_indices);
        free(local_scores);
        return score_fused_3bit_topk_range_6bit_f32(
            packed_db, fused6, indices_out, scores_out, 0, n_vectors, dim, bytes_per_vec, k
        );
    }

    int chunk = (n_vectors + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; t++) {
        int start = t * chunk;
        int end = start + chunk;
        if (start >= n_vectors) {
            break;
        }
        if (end > n_vectors) {
            end = n_vectors;
        }
        ctxs[t].packed_db = packed_db;
        ctxs[t].fused6 = fused6;
        ctxs[t].indices_out = local_indices + (size_t)t * k;
        ctxs[t].scores_out = local_scores + (size_t)t * k;
        ctxs[t].start = start;
        ctxs[t].end = end;
        ctxs[t].dim = dim;
        ctxs[t].bytes_per_vec = bytes_per_vec;
        ctxs[t].k = k;
        ctxs[t].found = 0;
        if (pthread_create(&threads[t], NULL, score_fused_3bit_topk_worker_6bit_f32, &ctxs[t]) != 0) {
            ctxs[t].found = score_fused_3bit_topk_range_6bit_f32(
                packed_db, fused6, ctxs[t].indices_out, ctxs[t].scores_out, start, end, dim, bytes_per_vec, k
            );
        } else {
            thread_created[t] = 1;
        }
    }

    for (int t = 0; t < num_threads; t++) {
        if (thread_created[t]) {
            pthread_join(threads[t], NULL);
        }
    }

    int found = 0;
    float min_score = 0.0f;
    int min_pos = 0;
    for (int t = 0; t < num_threads; t++) {
        int local_found = ctxs[t].found;
        int32_t *idxs = local_indices + (size_t)t * k;
        float *scores = local_scores + (size_t)t * k;
        for (int i = 0; i < local_found; i++) {
            float score = scores[i];
            if (found < k) {
                indices_out[found] = idxs[i];
                scores_out[found] = score;
                if (found == 0 || score < min_score) {
                    min_score = score;
                    min_pos = found;
                }
                found++;
                continue;
            }
            if (score <= min_score) {
                continue;
            }
            indices_out[min_pos] = idxs[i];
            scores_out[min_pos] = score;
            min_pos = 0;
            min_score = scores_out[0];
            for (int j = 1; j < k; j++) {
                if (scores_out[j] < min_score) {
                    min_score = scores_out[j];
                    min_pos = j;
                }
            }
        }
    }

    free(threads);
    free(thread_created);
    free(ctxs);
    free(local_indices);
    free(local_scores);

    sort_topk_desc(indices_out, scores_out, found);
    return found;
}

/* ================================================================
 * Weighted integer scorer — LUT-free affine-uniform 3-bit scoring
 * score = sum(weight[d] * level[d]) + bias
 * ================================================================ */

typedef struct {
    const uint8_t *packed_db;
    const float   *weights;
    float          bias;
    int32_t       *indices_out;
    float         *scores_out;
    int            start;
    int            end;
    int            dim;
    int            bytes_per_vec;
    int            k;
    int            found;
} topk_weighted_ctx_t;

static inline float score_row_3bit_weighted_fullgroups(
    const uint8_t *row,
    const float   *weights,
    int            n_groups)
{
    /*
     * Decode all 3-bit levels into a float buffer, then dot with weights.
     * The separation allows the compiler to auto-vectorize the dot product
     * (SSE/AVX on x86, NEON on ARM) via -O3 -march=native.
     */
    float levels[8];
    float total = 0.0f;
    const float *w = weights;

    for (int g = 0; g < n_groups; g++) {
        uint32_t u = (uint32_t)row[0] | ((uint32_t)row[1] << 8) | ((uint32_t)row[2] << 16);

        levels[0] = (float)(u & 7);
        levels[1] = (float)((u >> 3) & 7);
        levels[2] = (float)((u >> 6) & 7);
        levels[3] = (float)((u >> 9) & 7);
        levels[4] = (float)((u >> 12) & 7);
        levels[5] = (float)((u >> 15) & 7);
        levels[6] = (float)((u >> 18) & 7);
        levels[7] = (float)((u >> 21) & 7);

        total += w[0]*levels[0] + w[1]*levels[1] + w[2]*levels[2] + w[3]*levels[3]
               + w[4]*levels[4] + w[5]*levels[5] + w[6]*levels[6] + w[7]*levels[7];

        row += 3;
        w += 8;
    }

    return total;
}

static int score_3bit_weighted_topk_range(
    const uint8_t *packed_db,
    const float   *weights,
    float          bias,
    int32_t       *indices_out,
    float         *scores_out,
    int            start,
    int            end,
    int            dim,
    int            bytes_per_vec,
    int            k)
{
    const int n_groups = dim / 8;
    const int full_groups = (dim & 7) == 0;
    int found = 0;
    float min_score = 0.0f;
    int min_pos = 0;

    for (int v = start; v < end; v++) {
        const uint8_t *row = packed_db + (size_t)v * bytes_per_vec;
        float total;

        if (full_groups) {
            total = score_row_3bit_weighted_fullgroups(row, weights, n_groups) + bias;
        } else {
            /* General path for dim not divisible by 8 */
            total = bias;
            const float *w = weights;
            int bit_offset = 0;
            for (int d = 0; d < dim; d++) {
                int byte_idx = bit_offset >> 3;
                int shift = bit_offset & 7;
                uint32_t chunk = (uint32_t)row[byte_idx] >> shift;
                if (shift + 3 > 8) {
                    chunk |= (uint32_t)row[byte_idx + 1] << (8 - shift);
                }
                int level = chunk & 7;
                total += w[d] * (float)level;
                bit_offset += 3;
            }
        }

        if (found < k) {
            indices_out[found] = v;
            scores_out[found] = total;
            if (found == 0 || total < min_score) {
                min_score = total;
                min_pos = found;
            }
            found++;
            continue;
        }

        if (total <= min_score) {
            continue;
        }

        indices_out[min_pos] = v;
        scores_out[min_pos] = total;
        min_pos = 0;
        min_score = scores_out[0];
        for (int i = 1; i < k; i++) {
            if (scores_out[i] < min_score) {
                min_score = scores_out[i];
                min_pos = i;
            }
        }
    }

    sort_topk_desc(indices_out, scores_out, found);
    return found;
}

static void *score_3bit_weighted_topk_worker(void *arg)
{
    topk_weighted_ctx_t *ctx = (topk_weighted_ctx_t *)arg;
    ctx->found = score_3bit_weighted_topk_range(
        ctx->packed_db, ctx->weights, ctx->bias,
        ctx->indices_out, ctx->scores_out,
        ctx->start, ctx->end, ctx->dim, ctx->bytes_per_vec, ctx->k
    );
    return NULL;
}

int score_3bit_weighted_topk_dispatch(
    const uint8_t *packed_db,
    const float   *weights,
    float          bias,
    int32_t       *indices_out,
    float         *scores_out,
    int            n_vectors,
    int            dim,
    int            bytes_per_vec,
    int            k,
    int            num_threads)
{
    if (k <= 0) return 0;

    if (num_threads <= 1 || n_vectors < 32768) {
        return score_3bit_weighted_topk_range(
            packed_db, weights, bias, indices_out, scores_out,
            0, n_vectors, dim, bytes_per_vec, k
        );
    }

    if (num_threads > n_vectors) num_threads = n_vectors;

    pthread_t *threads = (pthread_t *)malloc((size_t)num_threads * sizeof(pthread_t));
    int *thread_created = (int *)calloc((size_t)num_threads, sizeof(int));
    topk_weighted_ctx_t *ctxs = (topk_weighted_ctx_t *)calloc((size_t)num_threads, sizeof(topk_weighted_ctx_t));
    int32_t *local_indices = (int32_t *)malloc((size_t)num_threads * (size_t)k * sizeof(int32_t));
    float *local_scores = (float *)malloc((size_t)num_threads * (size_t)k * sizeof(float));
    if (!threads || !thread_created || !ctxs || !local_indices || !local_scores) {
        free(threads); free(thread_created); free(ctxs);
        free(local_indices); free(local_scores);
        return score_3bit_weighted_topk_range(
            packed_db, weights, bias, indices_out, scores_out,
            0, n_vectors, dim, bytes_per_vec, k
        );
    }

    int chunk = (n_vectors + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; t++) {
        int start = t * chunk;
        int end_t = start + chunk;
        if (start >= n_vectors) break;
        if (end_t > n_vectors) end_t = n_vectors;
        ctxs[t].packed_db = packed_db;
        ctxs[t].weights = weights;
        ctxs[t].bias = bias;
        ctxs[t].indices_out = local_indices + (size_t)t * k;
        ctxs[t].scores_out = local_scores + (size_t)t * k;
        ctxs[t].start = start;
        ctxs[t].end = end_t;
        ctxs[t].dim = dim;
        ctxs[t].bytes_per_vec = bytes_per_vec;
        ctxs[t].k = k;
        ctxs[t].found = 0;
        if (pthread_create(&threads[t], NULL, score_3bit_weighted_topk_worker, &ctxs[t]) != 0) {
            ctxs[t].found = score_3bit_weighted_topk_range(
                packed_db, weights, bias, ctxs[t].indices_out, ctxs[t].scores_out,
                start, end_t, dim, bytes_per_vec, k
            );
        } else {
            thread_created[t] = 1;
        }
    }

    for (int t = 0; t < num_threads; t++) {
        if (thread_created[t]) pthread_join(threads[t], NULL);
    }

    /* Merge thread-local top-k results */
    int found = 0;
    float min_score = 0.0f;
    int min_pos = 0;
    for (int t = 0; t < num_threads; t++) {
        int local_found = ctxs[t].found;
        int32_t *idxs = local_indices + (size_t)t * k;
        float *scores = local_scores + (size_t)t * k;
        for (int i = 0; i < local_found; i++) {
            float score = scores[i];
            if (found < k) {
                indices_out[found] = idxs[i];
                scores_out[found] = score;
                if (found == 0 || score < min_score) {
                    min_score = score;
                    min_pos = found;
                }
                found++;
                continue;
            }
            if (score <= min_score) continue;
            indices_out[min_pos] = idxs[i];
            scores_out[min_pos] = score;
            min_pos = 0;
            min_score = scores_out[0];
            for (int j = 1; j < k; j++) {
                if (scores_out[j] < min_score) {
                    min_score = scores_out[j];
                    min_pos = j;
                }
            }
        }
    }

    free(threads); free(thread_created); free(ctxs);
    free(local_indices); free(local_scores);

    sort_topk_desc(indices_out, scores_out, found);
    return found;
}

/* ================================================================
 * Batch weighted scorer — decode each vector once, score all queries
 * ================================================================ */

typedef struct {
    const uint8_t *packed_db;
    const float   *weights_batch;  /* n_queries * dim */
    const float   *biases;         /* n_queries */
    int32_t       *indices_out;    /* n_queries * k */
    float         *scores_out;     /* n_queries * k */
    int           *found_out;      /* n_queries */
    int            start;
    int            end;
    int            dim;
    int            bytes_per_vec;
    int            k;
    int            n_queries;
} topk_weighted_batch_ctx_t;

static void score_3bit_weighted_batch_topk_range(
    const uint8_t *packed_db,
    const float   *weights_batch,
    const float   *biases,
    int32_t       *indices_out,
    float         *scores_out,
    int           *found_out,
    int            start,
    int            end,
    int            dim,
    int            bytes_per_vec,
    int            k,
    int            n_queries)
{
    const int n_groups = dim / 8;
    const int full_groups = (dim & 7) == 0;

    /* Per-query top-k state */
    float *min_scores = (float *)calloc((size_t)n_queries, sizeof(float));
    int *min_positions = (int *)calloc((size_t)n_queries, sizeof(int));
    if (!min_scores || !min_positions) {
        free(min_scores);
        free(min_positions);
        return;
    }
    memset(found_out, 0, (size_t)n_queries * sizeof(int));

    /* Temporary buffer for decoded levels (one vector at a time) */
    float *levels = (float *)malloc((size_t)dim * sizeof(float));
    if (!levels) {
        free(min_scores);
        free(min_positions);
        return;
    }

    for (int v = start; v < end; v++) {
        const uint8_t *row = packed_db + (size_t)v * bytes_per_vec;

        /* Decode 3-bit levels from packed bytes — done ONCE per vector */
        if (full_groups) {
            const uint8_t *r = row;
            float *lv = levels;
            for (int g = 0; g < n_groups; g++) {
                uint32_t u = (uint32_t)r[0] | ((uint32_t)r[1] << 8) | ((uint32_t)r[2] << 16);
                lv[0] = (float)(u & 7);
                lv[1] = (float)((u >> 3) & 7);
                lv[2] = (float)((u >> 6) & 7);
                lv[3] = (float)((u >> 9) & 7);
                lv[4] = (float)((u >> 12) & 7);
                lv[5] = (float)((u >> 15) & 7);
                lv[6] = (float)((u >> 18) & 7);
                lv[7] = (float)((u >> 21) & 7);
                r += 3;
                lv += 8;
            }
        } else {
            int bit_offset = 0;
            for (int d = 0; d < dim; d++) {
                int byte_idx = bit_offset >> 3;
                int shift = bit_offset & 7;
                uint32_t chunk = (uint32_t)row[byte_idx] >> shift;
                if (shift + 3 > 8) {
                    chunk |= (uint32_t)row[byte_idx + 1] << (8 - shift);
                }
                levels[d] = (float)(chunk & 7);
                bit_offset += 3;
            }
        }

        /* Score against all queries using decoded levels */
        for (int q = 0; q < n_queries; q++) {
            const float *w = weights_batch + (size_t)q * dim;
            float score = biases[q];
            for (int d = 0; d < dim; d++) {
                score += w[d] * levels[d];
            }

            int32_t *q_indices = indices_out + (size_t)q * k;
            float *q_scores = scores_out + (size_t)q * k;
            int qfound = found_out[q];

            if (qfound < k) {
                q_indices[qfound] = v;
                q_scores[qfound] = score;
                if (qfound == 0 || score < min_scores[q]) {
                    min_scores[q] = score;
                    min_positions[q] = qfound;
                }
                found_out[q] = qfound + 1;
                continue;
            }

            if (score <= min_scores[q]) continue;

            q_indices[min_positions[q]] = v;
            q_scores[min_positions[q]] = score;
            min_positions[q] = 0;
            min_scores[q] = q_scores[0];
            for (int i = 1; i < k; i++) {
                if (q_scores[i] < min_scores[q]) {
                    min_scores[q] = q_scores[i];
                    min_positions[q] = i;
                }
            }
        }
    }

    /* Sort each query's results */
    for (int q = 0; q < n_queries; q++) {
        sort_topk_desc(
            indices_out + (size_t)q * k,
            scores_out + (size_t)q * k,
            found_out[q]
        );
    }

    free(levels);
    free(min_scores);
    free(min_positions);
}

static void *score_3bit_weighted_batch_topk_worker(void *arg)
{
    topk_weighted_batch_ctx_t *ctx = (topk_weighted_batch_ctx_t *)arg;
    score_3bit_weighted_batch_topk_range(
        ctx->packed_db, ctx->weights_batch, ctx->biases,
        ctx->indices_out, ctx->scores_out, ctx->found_out,
        ctx->start, ctx->end, ctx->dim, ctx->bytes_per_vec,
        ctx->k, ctx->n_queries
    );
    return NULL;
}

int score_3bit_weighted_batch_topk_dispatch(
    const uint8_t *packed_db,
    const float   *weights_batch,
    const float   *biases,
    int32_t       *indices_out,
    float         *scores_out,
    int           *found_out,
    int            n_vectors,
    int            dim,
    int            bytes_per_vec,
    int            k,
    int            n_queries,
    int            num_threads)
{
    if (k <= 0 || n_queries <= 0) return 0;

    if (num_threads <= 1 || n_vectors < 32768) {
        score_3bit_weighted_batch_topk_range(
            packed_db, weights_batch, biases,
            indices_out, scores_out, found_out,
            0, n_vectors, dim, bytes_per_vec, k, n_queries
        );
        return 0;
    }

    if (num_threads > n_vectors) num_threads = n_vectors;

    pthread_t *threads = (pthread_t *)malloc((size_t)num_threads * sizeof(pthread_t));
    int *thread_created = (int *)calloc((size_t)num_threads, sizeof(int));
    topk_weighted_batch_ctx_t *ctxs = (topk_weighted_batch_ctx_t *)calloc((size_t)num_threads, sizeof(topk_weighted_batch_ctx_t));
    /* Each thread needs n_queries * k entries for indices, scores, found */
    size_t per_thread = (size_t)n_queries * (size_t)k;
    int32_t *local_indices = (int32_t *)malloc((size_t)num_threads * per_thread * sizeof(int32_t));
    float *local_scores = (float *)malloc((size_t)num_threads * per_thread * sizeof(float));
    int *local_found = (int *)calloc((size_t)num_threads * (size_t)n_queries, sizeof(int));

    if (!threads || !thread_created || !ctxs || !local_indices || !local_scores || !local_found) {
        free(threads); free(thread_created); free(ctxs);
        free(local_indices); free(local_scores); free(local_found);
        score_3bit_weighted_batch_topk_range(
            packed_db, weights_batch, biases,
            indices_out, scores_out, found_out,
            0, n_vectors, dim, bytes_per_vec, k, n_queries
        );
        return 0;
    }

    int chunk = (n_vectors + num_threads - 1) / num_threads;
    int active_threads = 0;
    for (int t = 0; t < num_threads; t++) {
        int start = t * chunk;
        int end_t = start + chunk;
        if (start >= n_vectors) break;
        if (end_t > n_vectors) end_t = n_vectors;
        ctxs[t].packed_db = packed_db;
        ctxs[t].weights_batch = weights_batch;
        ctxs[t].biases = biases;
        ctxs[t].indices_out = local_indices + (size_t)t * per_thread;
        ctxs[t].scores_out = local_scores + (size_t)t * per_thread;
        ctxs[t].found_out = local_found + (size_t)t * n_queries;
        ctxs[t].start = start;
        ctxs[t].end = end_t;
        ctxs[t].dim = dim;
        ctxs[t].bytes_per_vec = bytes_per_vec;
        ctxs[t].k = k;
        ctxs[t].n_queries = n_queries;
        if (pthread_create(&threads[t], NULL, score_3bit_weighted_batch_topk_worker, &ctxs[t]) != 0) {
            score_3bit_weighted_batch_topk_range(
                packed_db, weights_batch, biases,
                ctxs[t].indices_out, ctxs[t].scores_out, ctxs[t].found_out,
                start, end_t, dim, bytes_per_vec, k, n_queries
            );
        } else {
            thread_created[t] = 1;
        }
        active_threads = t + 1;
    }

    for (int t = 0; t < active_threads; t++) {
        if (thread_created[t]) pthread_join(threads[t], NULL);
    }

    /* Merge per-thread results for each query */
    memset(found_out, 0, (size_t)n_queries * sizeof(int));
    for (int q = 0; q < n_queries; q++) {
        int32_t *q_out_idx = indices_out + (size_t)q * k;
        float *q_out_scores = scores_out + (size_t)q * k;
        int found = 0;
        float min_s = 0.0f;
        int min_p = 0;

        for (int t = 0; t < active_threads; t++) {
            int lf = local_found[(size_t)t * n_queries + q];
            int32_t *lidx = local_indices + (size_t)t * per_thread + (size_t)q * k;
            float *lscr = local_scores + (size_t)t * per_thread + (size_t)q * k;
            for (int i = 0; i < lf; i++) {
                float score = lscr[i];
                if (found < k) {
                    q_out_idx[found] = lidx[i];
                    q_out_scores[found] = score;
                    if (found == 0 || score < min_s) {
                        min_s = score;
                        min_p = found;
                    }
                    found++;
                    continue;
                }
                if (score <= min_s) continue;
                q_out_idx[min_p] = lidx[i];
                q_out_scores[min_p] = score;
                min_p = 0;
                min_s = q_out_scores[0];
                for (int j = 1; j < k; j++) {
                    if (q_out_scores[j] < min_s) {
                        min_s = q_out_scores[j];
                        min_p = j;
                    }
                }
            }
        }
        found_out[q] = found;
        sort_topk_desc(q_out_idx, q_out_scores, found);
    }

    free(threads); free(thread_created); free(ctxs);
    free(local_indices); free(local_scores); free(local_found);

    return 0;
}
