#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
// GGML CPU internal header

#ifdef __cplusplus
extern "C" {
#endif

// Quantization
void quantize_row_q4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q4_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

void quantize_row_q2_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q3_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q4_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q6_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

void quantize_row_tq1_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_tq2_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

void quantize_row_iq4_nl (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_iq4_xs (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

// Dot product
void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q8_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_q2_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q3_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_tq1_0_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_tq2_0_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_iq2_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_xs_q8_K (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq1_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq1_m_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_nl_q8_0 (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_xs_q8_K (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

// mmla
static size_t g_chunk_size_q8_0_mmla_spack = 1024;
typedef struct matmul_q8_0_mmla_spack_t
{
    // 每2个A行混部8个
    // 每4个B行混部8个
    int8_t* p;
    size_t p_stride;
    // 每2*32=64个pa有2个scale
    // 每4*32=128个pb有4个scale
    float32_t* scale;
    size_t scale_stride;
    size_t sz_p;
    size_t sz_scale;
} matmul_q8_0_mmla_spack_t;

typedef void (*matmul_q8_0_mmla_spack_func)(size_t M,
                                            size_t N,
                                            size_t K,
                                            const matmul_q8_0_mmla_spack_t* pack_a,
                                            const matmul_q8_0_mmla_spack_t* pack_b,
                                            void* GGML_RESTRICT c,
                                            size_t c_column_bytes);
void
matmul_q8_0_mmla_spack_b(size_t na,
                         size_t nb,
                         int dim,
                         const void* GGML_RESTRICT a,
                         const void* GGML_RESTRICT b,
                         void* GGML_RESTRICT c,
                         size_t c_column_bytes,
                         matmul_q8_0_mmla_spack_func func);


void
matmul_q8_0_mmla_4x4_b_micro_spack(size_t M,
                                        size_t N,
                                        size_t K,
                                        const matmul_q8_0_mmla_spack_t * pack_a,
                                        const matmul_q8_0_mmla_spack_t * pack_b,
                                        void * GGML_RESTRICT c,
                                        size_t c_column_bytes);
void
matmul_q8_0_mmla_4x4_b_micro_spack_neon(size_t M,
                                        size_t N,
                                        size_t K,
                                        const matmul_q8_0_mmla_spack_t * pack_a,
                                        const matmul_q8_0_mmla_spack_t * pack_b,
                                        void * GGML_RESTRICT c,
                                        size_t c_column_bytes);
void
matmul_q8_0_mmla_8x8_b_micro_spack(size_t M,
                                   size_t N,
                                   size_t K,
                                   const matmul_q8_0_mmla_spack_t* pack_a,
                                   const matmul_q8_0_mmla_spack_t* pack_b,
                                   void* GGML_RESTRICT c,
                                   size_t c_column_bytes);
void
matmul_q8_0_mmla_8x8_b_micro_spack_neon(size_t M,
                                   size_t N,
                                   size_t K,
                                   const matmul_q8_0_mmla_spack_t* pack_a,
                                   const matmul_q8_0_mmla_spack_t* pack_b,
                                   void* GGML_RESTRICT c,
                                   size_t c_column_bytes);

void 
matmul_q8_0_mmla_mask_b_micro_spack(size_t M,
                                         size_t N,
                                         size_t K,
                                         const matmul_q8_0_mmla_spack_t * pack_a,
                                         const matmul_q8_0_mmla_spack_t * pack_b,
                                         void * GGML_RESTRICT c,
                                         size_t c_column_bytes);


void
matmul_q8_0_mmla_spack_padded(bool isA,
                              size_t nr_real,
                              size_t nr_pack,
                              int dim,
                              const block_q8_0* GGML_RESTRICT X,
                              size_t column_bytes,
                              matmul_q8_0_mmla_spack_t** out);
void
matmul_q8_0_mmla_4x4_tiled_padded(size_t na_real,
                                  size_t nb_real,
                                  size_t na_pad,
                                  size_t nb_pad,
                                  size_t k,
                                  size_t m,
                                  size_t n,
                                  size_t kb,
                                  const block_q8_0 * GGML_RESTRICT A,
                                  const block_q8_0 * GGML_RESTRICT B,
                                  size_t a_stride_q_0,
                                  size_t b_stride_q_0,
                                  size_t a_column_bytes,
                                  size_t b_column_bytes,
                                  float * GGML_RESTRICT tmp_c,
                                  matmul_q8_0_mmla_spack_t ** pack_a_tile_io,
                                  matmul_q8_0_mmla_spack_t ** pack_b_tile_io);

static inline void
matmul_q8_0_mmla_spack_b_g(size_t na,
                           size_t nb,
                           int dim,
                           const void* GGML_RESTRICT a,
                           const void* GGML_RESTRICT b,
                           void* GGML_RESTRICT c,
                           size_t c_column_bytes)
{
    matmul_q8_0_mmla_spack_b(na, nb, dim, a, b, c, c_column_bytes, matmul_q8_0_mmla_8x8_b_micro_spack_neon);
}

#ifdef __cplusplus
}
#endif
