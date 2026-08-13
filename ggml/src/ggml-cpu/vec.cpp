#include "vec.h"

#include <cassert>

// Suppress pre-existing warnings not from our changes
#pragma GCC diagnostic ignored "-Wmissing-declarations"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wvla"
#pragma GCC diagnostic ignored "-Wunused-parameter"

// precomputed gelu table for f16 (128 KB)
ggml_fp16_t ggml_table_gelu_f16[1 << 16];

// precomputed quick gelu table for f16 (128 KB)
ggml_fp16_t ggml_table_gelu_quick_f16[1 << 16];

void ggml_vec_dot_f32(int n, float * GGML_RESTRICT s, size_t bs, const float * GGML_RESTRICT x, size_t bx, const float * GGML_RESTRICT y, size_t by, int nrc) {
   assert(nrc == 1);
   GGML_UNUSED(nrc);
   GGML_UNUSED(bx);
   GGML_UNUSED(by);
   GGML_UNUSED(bs);

#if defined(GGML_SIMD)
    float sumf = 0.0f;

    #if defined(__ARM_FEATURE_SVE)
        const int sve_register_length = ggml_cpu_get_sve_cnt() * 8;
        const int ggml_f32_epr = sve_register_length / 32;//8;//svcntw(); // SVE128:4, SVE256:8, SVE512:16
        const int ggml_f32_step = 8 * ggml_f32_epr; // choose 8 SVE registers

        const int np = (n & ~(ggml_f32_step - 1));
        svfloat32_t sum1 = svdup_n_f32(0.0f);
        svfloat32_t sum2 = svdup_n_f32(0.0f);
        svfloat32_t sum3 = svdup_n_f32(0.0f);
        svfloat32_t sum4 = svdup_n_f32(0.0f);
        svfloat32_t sum5 = svdup_n_f32(0.0f);
        svfloat32_t sum6 = svdup_n_f32(0.0f);
        svfloat32_t sum7 = svdup_n_f32(0.0f);
        svfloat32_t sum8 = svdup_n_f32(0.0f);
        svfloat32_t ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8;
        svfloat32_t ay1,ay2,ay3,ay4,ay5,ay6,ay7,ay8;
        for (int i = 0; i < np; i += ggml_f32_step) {
            ax1 = GGML_F32_VEC_LOAD(x + i);
            ay1 = GGML_F32_VEC_LOAD(y + i);
            sum1 = GGML_F32_VEC_FMA(ax1, ay1, sum1);

            ax2 = GGML_F32_VEC_LOAD(x + i + 1*ggml_f32_epr);
            ay2 = GGML_F32_VEC_LOAD(y + i + 1*ggml_f32_epr);
            sum2 = GGML_F32_VEC_FMA(ax2, ay2, sum2);

            ax3 = GGML_F32_VEC_LOAD(x + i + 2*ggml_f32_epr);
            ay3 = GGML_F32_VEC_LOAD(y + i + 2*ggml_f32_epr);
            sum3 = GGML_F32_VEC_FMA(ax3, ay3, sum3);

            ax4 = GGML_F32_VEC_LOAD(x + i + 3*ggml_f32_epr);
            ay4 = GGML_F32_VEC_LOAD(y + i + 3*ggml_f32_epr);
            sum4 = GGML_F32_VEC_FMA(ax4, ay4, sum4);

            ax5 = GGML_F32_VEC_LOAD(x + i + 4*ggml_f32_epr);
            ay5 = GGML_F32_VEC_LOAD(y + i + 4*ggml_f32_epr);
            sum5 = GGML_F32_VEC_FMA(ax5, ay5, sum5);

            ax6 = GGML_F32_VEC_LOAD(x + i + 5*ggml_f32_epr);
            ay6 = GGML_F32_VEC_LOAD(y + i + 5*ggml_f32_epr);
            sum6 = GGML_F32_VEC_FMA(ax6, ay6, sum6);

            ax7 = GGML_F32_VEC_LOAD(x + i + 6*ggml_f32_epr);
            ay7 = GGML_F32_VEC_LOAD(y + i + 6*ggml_f32_epr);
            sum7 = GGML_F32_VEC_FMA(ax7, ay7, sum7);

            ax8 = GGML_F32_VEC_LOAD(x + i + 7*ggml_f32_epr);
            ay8 = GGML_F32_VEC_LOAD(y + i + 7*ggml_f32_epr);
            sum8 = GGML_F32_VEC_FMA(ax8, ay8, sum8);
        }
        // leftovers
        // Since 8 unrolls are done in above loop, leftovers lie in range [0, ggml_f32_step] which is handled in below loop
        const int np2 = (n & ~(ggml_f32_epr - 1));
        for (int i = np; i < np2; i += ggml_f32_epr) {
            ax1 = GGML_F32_VEC_LOAD(x + i);
            ay1 = GGML_F32_VEC_LOAD(y + i);
            sum1 = GGML_F32_VEC_FMA(ax1, ay1, sum1);
        }
        // maximum number of leftover elements will be less that ggml_f32_epr. Apply predicated svmla on available elements only
        // Note: must use svmla_f32_m (not svmad_f32_m) so that inactive lanes preserve sum1 (=op1, the destination)
        // rather than op2 (which is 0 from predicated svld1). Otherwise the partial tail corrupts accumulator lanes.
        if (np2 < n) {
            svbool_t pg = svwhilelt_b32(np2, n);
            ax1 = svld1_f32(pg, x + np2);
            ay1 = svld1_f32(pg, y + np2);
            sum1 = svmla_f32_m(pg, sum1, ax1, ay1);
        }
        // reduce sum1,sum2 to sum1
        GGML_F32_VEC_REDUCE(sumf, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8);
    #else
        const int np = (n & ~(GGML_F32_STEP - 1));

        GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

        GGML_F32_VEC ax[GGML_F32_ARR];
        GGML_F32_VEC ay[GGML_F32_ARR];

        for (int i = 0; i < np; i += GGML_F32_STEP) {
            for (int j = 0; j < GGML_F32_ARR; j++) {
                ax[j] = GGML_F32_VEC_LOAD(x + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);

                sum[j] = GGML_F32_VEC_FMA(sum[j], ax[j], ay[j]);
            }
        }

        // reduce sum0..sum3 to sum0
        GGML_F32_VEC_REDUCE(sumf, sum);

        // leftovers
        for (int i = np; i < n; ++i) {
            sumf += x[i]*y[i];
        }
    #endif
#else
    // scalar
    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float)(x[i]*y[i]);
    }
#endif

    *s = sumf;
}

void ggml_vec_dot_bf16(int n, float * GGML_RESTRICT s, size_t bs, ggml_bf16_t * GGML_RESTRICT x, size_t bx, ggml_bf16_t * GGML_RESTRICT y, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
    int i = 0;
    ggml_float sumf = 0;

#if defined(__AVX512BF16__)
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    for (; i + 64 <= n; i += 64) {
        c1 = _mm512_dpbf16_ps(c1, m512bh(_mm512_loadu_si512((x + i))),
                             m512bh(_mm512_loadu_si512((y + i))));
        c2 = _mm512_dpbf16_ps(c2, m512bh(_mm512_loadu_si512((x + i + 32))),
                             m512bh(_mm512_loadu_si512((y + i + 32))));
    }
    sumf += (ggml_float)_mm512_reduce_add_ps(c1);
    sumf += (ggml_float)_mm512_reduce_add_ps(c2);

#elif defined(__AVX512F__)
#define LOAD(p) _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(p))), 16))
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        c1 = _mm512_add_ps(_mm512_mul_ps(LOAD(x + i), LOAD(y + i)), c1);
        c2 = _mm512_add_ps(_mm512_mul_ps(LOAD(x + i + 16), LOAD(y + i + 16)), c2);
    }
    sumf += (ggml_float)_mm512_reduce_add_ps(c1);
    sumf += (ggml_float)_mm512_reduce_add_ps(c2);

#undef LOAD
#elif defined(__AVX2__) || defined(__AVX__)
#if defined(__AVX2__)
#define LOAD(p) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p))), 16))
#else
#define LOAD(p) _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p))), 16)), (_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_bsrli_si128(_mm_loadu_si128((const __m128i *)(p)), 8)), 16)), 1))
#endif
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        c1 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i), LOAD(y + i)), c1);
        c2 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 8), LOAD(y + i + 8)), c2);
        c3 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 16), LOAD(y + i + 16)), c3);
        c4 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 24), LOAD(y + i + 24)), c4);
    }
    __m128 g;
    c1 = _mm256_add_ps(_mm256_add_ps(c1, c3),
                       _mm256_add_ps(c2, c4));
    g = _mm_add_ps(_mm256_extractf128_ps(c1, 1),
                   _mm256_castps256_ps128(c1));
    g = _mm_add_ps(g, _mm_movehl_ps(g, g));
    g = _mm_add_ss(g, _mm_movehdup_ps(g));
    sumf += (ggml_float)_mm_cvtss_f32(g);

#undef LOAD
#endif

    for (; i < n; ++i) {
        sumf += (ggml_float)(GGML_BF16_TO_FP32(x[i]) *
                             GGML_BF16_TO_FP32(y[i]));
    }
    *s = sumf;
}

void ggml_vec_dot_f16(int n, float * GGML_RESTRICT s, size_t bs, ggml_fp16_t * GGML_RESTRICT x, size_t bx, ggml_fp16_t * GGML_RESTRICT y, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    // NEON optimized path: process 64 fp16 elements at a time with 8 NEON registers
    // Software pipeline to hide load latency
    const int np = (n & ~((1 << 6) - 1)); // 64-element aligned

    float16x8_t acc0 = vdupq_n_f16(0);
    float16x8_t acc1 = vdupq_n_f16(0);
    float16x8_t acc2 = vdupq_n_f16(0);
    float16x8_t acc3 = vdupq_n_f16(0);
    float16x8_t acc4 = vdupq_n_f16(0);
    float16x8_t acc5 = vdupq_n_f16(0);
    float16x8_t acc6 = vdupq_n_f16(0);
    float16x8_t acc7 = vdupq_n_f16(0);

    // Pre-load first 64 elements
    float16x8_t ax0 = vld1q_f16((const __fp16 *)(x + 0));
    float16x8_t ax1 = vld1q_f16((const __fp16 *)(x + 8));
    float16x8_t ax2 = vld1q_f16((const __fp16 *)(x + 16));
    float16x8_t ax3 = vld1q_f16((const __fp16 *)(x + 24));
    float16x8_t ax4 = vld1q_f16((const __fp16 *)(x + 32));
    float16x8_t ax5 = vld1q_f16((const __fp16 *)(x + 40));
    float16x8_t ax6 = vld1q_f16((const __fp16 *)(x + 48));
    float16x8_t ax7 = vld1q_f16((const __fp16 *)(x + 56));

    float16x8_t ay0 = vld1q_f16((const __fp16 *)(y + 0));
    float16x8_t ay1 = vld1q_f16((const __fp16 *)(y + 8));
    float16x8_t ay2 = vld1q_f16((const __fp16 *)(y + 16));
    float16x8_t ay3 = vld1q_f16((const __fp16 *)(y + 24));
    float16x8_t ay4 = vld1q_f16((const __fp16 *)(y + 32));
    float16x8_t ay5 = vld1q_f16((const __fp16 *)(y + 40));
    float16x8_t ay6 = vld1q_f16((const __fp16 *)(y + 48));
    float16x8_t ay7 = vld1q_f16((const __fp16 *)(y + 56));

    int i = 64;
    for (; i < np; i += 64) {
        // Software pipeline: compute with previous data, load next
        acc0 = vfmaq_f16(acc0, ax0, ay0);
        ax0 = vld1q_f16((const __fp16 *)(x + i + 0));
        ay0 = vld1q_f16((const __fp16 *)(y + i + 0));

        acc1 = vfmaq_f16(acc1, ax1, ay1);
        ax1 = vld1q_f16((const __fp16 *)(x + i + 8));
        ay1 = vld1q_f16((const __fp16 *)(y + i + 8));

        acc2 = vfmaq_f16(acc2, ax2, ay2);
        ax2 = vld1q_f16((const __fp16 *)(x + i + 16));
        ay2 = vld1q_f16((const __fp16 *)(y + i + 16));

        acc3 = vfmaq_f16(acc3, ax3, ay3);
        ax3 = vld1q_f16((const __fp16 *)(x + i + 24));
        ay3 = vld1q_f16((const __fp16 *)(y + i + 24));

        acc4 = vfmaq_f16(acc4, ax4, ay4);
        ax4 = vld1q_f16((const __fp16 *)(x + i + 32));
        ay4 = vld1q_f16((const __fp16 *)(y + i + 32));

        acc5 = vfmaq_f16(acc5, ax5, ay5);
        ax5 = vld1q_f16((const __fp16 *)(x + i + 40));
        ay5 = vld1q_f16((const __fp16 *)(y + i + 40));

        acc6 = vfmaq_f16(acc6, ax6, ay6);
        ax6 = vld1q_f16((const __fp16 *)(x + i + 48));
        ay6 = vld1q_f16((const __fp16 *)(y + i + 48));

        acc7 = vfmaq_f16(acc7, ax7, ay7);
        ax7 = vld1q_f16((const __fp16 *)(x + i + 56));
        ay7 = vld1q_f16((const __fp16 *)(y + i + 56));
    }

    // Epilogue: process last block (already loaded, no more loads)
    acc0 = vfmaq_f16(acc0, ax0, ay0);
    acc1 = vfmaq_f16(acc1, ax1, ay1);
    acc2 = vfmaq_f16(acc2, ax2, ay2);
    acc3 = vfmaq_f16(acc3, ax3, ay3);
    acc4 = vfmaq_f16(acc4, ax4, ay4);
    acc5 = vfmaq_f16(acc5, ax5, ay5);
    acc6 = vfmaq_f16(acc6, ax6, ay6);
    acc7 = vfmaq_f16(acc7, ax7, ay7);

    // Horizontal sum: sum all 8 acc vectors into one scalar
    float16x8_t sum = vpaddq_f16(acc0, acc1);
    float16x8_t sum2 = vpaddq_f16(acc2, acc3);
    float16x8_t sum3 = vpaddq_f16(acc4, acc5);
    float16x8_t sum4 = vpaddq_f16(acc6, acc7);
    sum = vpaddq_f16(sum, sum2);
    sum3 = vpaddq_f16(sum3, sum4);
    sum = vpaddq_f16(sum, sum3);
    // Now sum has 8 partial sums, need one more step
    float32_t r0 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(sum))) +
                   vaddvq_f32(vcvt_f32_f16(vget_high_f16(sum)));
    ggml_float sumf = (ggml_float)r0;

    // leftovers (0-63 elements)
    for (int i2 = np; i2 < n; ++i2) {
        sumf += (ggml_float)(GGML_FP16_TO_FP32(x[i2])*GGML_FP16_TO_FP32(y[i2]));
    }

    *s = (float)sumf;
}

float16x8_t svget_neonq_f16(svfloat16_t sve_vec) {
    // return svget_neonq(sve_vec);
    float16_t v[svcnth()];
    svst1(svptrue_b16(), v, sve_vec);
    return vld1q_f16(v);
}

void ggml_matmul_f32_4x4_kernel(size_t na, size_t nb, int dim,
               const void* GGML_RESTRICT a, const void* GGML_RESTRICT b,
               void* GGML_RESTRICT c, size_t c_column_bytes)
{
    // fprintf(stderr, "%s(na=%lu, nb=%lu, dim=%d)\n", __func__, na, nb, dim);
    const size_t M = na;
    const size_t N = nb;
    const size_t K = dim;
    const float32_t* GGML_RESTRICT A = (float32_t*)(a);
    const float32_t* GGML_RESTRICT B = (float32_t*)(b);
    float32_t* GGML_RESTRICT C = (float32_t*)(c);
    size_t c_stride = c_column_bytes / sizeof(float32_t);
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            // for (int k = 0; k < dim; k++) {
            //     C[i * c_stride + j] += A[j * dim + k] * B[i * dim + k];
            // }
            ggml_vec_dot_f32(dim, &C[(i * c_stride) + j], 0, &A[j * dim], 0, &B[i * dim], 0, 1);
        }
    }
    return;
    const size_t vl = svcntw();
    svbool_t pg_all = svptrue_b8();
    (void)pg_all;
    (void)vl;
    size_t N_rounded = N & ~((1 << 2) - 1);
    size_t M_rounded = M & ~((1 << 2) - 1);
    svfloat32_t acc00, acc01, acc02, acc03, acc10, acc11, acc12, acc13, acc20, acc21, acc22, acc23, acc30, acc31, acc32,
      acc33;
    // const float zero_f32_x[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    svfloat32_t zero = svdup_f32(0);
    for (size_t i = 0; i < N_rounded; i += 4) {
        for (size_t j = 0; j < M_rounded; j += 4) {
// #define LOAD_ZERO(acc) asm volatile("ld1w {%0.d}, %2/z, [%1]\n\t" : "=w"(acc) : "r"(zero_f32_x), "Upl"(pg_all) :);
#define LOAD_ZERO(acc) (acc) = zero;
            LOAD_ZERO(acc00);
            LOAD_ZERO(acc01);
            LOAD_ZERO(acc02);
            LOAD_ZERO(acc03);
            LOAD_ZERO(acc10);
            LOAD_ZERO(acc11);
            LOAD_ZERO(acc12);
            LOAD_ZERO(acc13);
            LOAD_ZERO(acc20);
            LOAD_ZERO(acc21);
            LOAD_ZERO(acc22);
            LOAD_ZERO(acc23);
            LOAD_ZERO(acc30);
            LOAD_ZERO(acc31);
            LOAD_ZERO(acc32);
            LOAD_ZERO(acc33);
#undef LOAD_ZERO
            for (size_t k = 0; k < K; k += vl) {
                svbool_t pg = svwhilelt_b32(k, K);
                svfloat32_t j0 = svld1_f32(pg, &A[(j + 0) * K + k]);
                svfloat32_t j1 = svld1_f32(pg, &A[(j + 1) * K + k]);
                svfloat32_t j2 = svld1_f32(pg, &A[(j + 2) * K + k]);
                svfloat32_t j3 = svld1_f32(pg, &A[(j + 3) * K + k]);
                svfloat32_t i0 = svld1_f32(pg, &B[(i + 0) * K + k]);
                svfloat32_t i1 = svld1_f32(pg, &B[(i + 1) * K + k]);
                svfloat32_t i2 = svld1_f32(pg, &B[(i + 2) * K + k]);
                svfloat32_t i3 = svld1_f32(pg, &B[(i + 3) * K + k]);
                acc00 = svmla_f32_m(pg, acc00, i0, j0);
                acc01 = svmla_f32_m(pg, acc01, i0, j1);
                acc02 = svmla_f32_m(pg, acc02, i0, j2);
                acc03 = svmla_f32_m(pg, acc03, i0, j3);
                acc10 = svmla_f32_m(pg, acc10, i1, j0);
                acc11 = svmla_f32_m(pg, acc11, i1, j1);
                acc12 = svmla_f32_m(pg, acc12, i1, j2);
                acc13 = svmla_f32_m(pg, acc13, i1, j3);
                acc20 = svmla_f32_m(pg, acc20, i2, j0);
                acc21 = svmla_f32_m(pg, acc21, i2, j1);
                acc22 = svmla_f32_m(pg, acc22, i2, j2);
                acc23 = svmla_f32_m(pg, acc23, i2, j3);
                acc30 = svmla_f32_m(pg, acc30, i3, j0);
                acc31 = svmla_f32_m(pg, acc31, i3, j1);
                acc32 = svmla_f32_m(pg, acc32, i3, j2);
                acc33 = svmla_f32_m(pg, acc33, i3, j3);
            }
            svbool_t pg_all = svptrue_b16();
            float32_t r00 = svaddv_f32(pg_all, acc00);
            float32_t r01 = svaddv_f32(pg_all, acc01);
            float32_t r02 = svaddv_f32(pg_all, acc02);
            float32_t r03 = svaddv_f32(pg_all, acc03);
            float32_t r10 = svaddv_f32(pg_all, acc10);
            float32_t r11 = svaddv_f32(pg_all, acc11);
            float32_t r12 = svaddv_f32(pg_all, acc12);
            float32_t r13 = svaddv_f32(pg_all, acc13);
            float32_t r20 = svaddv_f32(pg_all, acc20);
            float32_t r21 = svaddv_f32(pg_all, acc21);
            float32_t r22 = svaddv_f32(pg_all, acc22);
            float32_t r23 = svaddv_f32(pg_all, acc23);
            float32_t r30 = svaddv_f32(pg_all, acc30);
            float32_t r31 = svaddv_f32(pg_all, acc31);
            float32_t r32 = svaddv_f32(pg_all, acc32);
            float32_t r33 = svaddv_f32(pg_all, acc33);
            C[((i + 0) * c_stride) + (j + 0)] = r00;
            C[((i + 0) * c_stride) + (j + 1)] = r01;
            C[((i + 0) * c_stride) + (j + 2)] = r02;
            C[((i + 0) * c_stride) + (j + 3)] = r03;
            C[((i + 1) * c_stride) + (j + 0)] = r10;
            C[((i + 1) * c_stride) + (j + 1)] = r11;
            C[((i + 1) * c_stride) + (j + 2)] = r12;
            C[((i + 1) * c_stride) + (j + 3)] = r13;
            C[((i + 2) * c_stride) + (j + 0)] = r20;
            C[((i + 2) * c_stride) + (j + 1)] = r21;
            C[((i + 2) * c_stride) + (j + 2)] = r22;
            C[((i + 2) * c_stride) + (j + 3)] = r23;
            C[((i + 3) * c_stride) + (j + 0)] = r30;
            C[((i + 3) * c_stride) + (j + 1)] = r31;
            C[((i + 3) * c_stride) + (j + 2)] = r32;
            C[((i + 3) * c_stride) + (j + 3)] = r33;
        }
    }
    // Handle leftovers with dot
    for (size_t i = 0; i < N; i++) {
        for (size_t j = M_rounded; j < M; j++) {
            ggml_vec_dot_f32(dim, &C[(i * c_stride) + j], 0, &A[j * dim], 0, &B[i * dim], 0, 1);
        }
    }
    for (size_t i = N_rounded; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            ggml_vec_dot_f32(dim, &C[(i * c_stride) + j], 0, &A[j * dim], 0, &B[i * dim], 0, 1);
        }
    }
}

// Performance-first SVE 1x8 kernel for fixed 256-bit SVE
// Assumption: SVE vector length = 256-bit = 16 fp16 lanes
// M == 1 decode fast path
static inline void
ggml_matmul_f16_1xn_kernel_sve(size_t N, size_t K,
                               const ggml_fp16_t * GGML_RESTRICT A,
                               const ggml_fp16_t * GGML_RESTRICT B,
                               float32_t * GGML_RESTRICT C,
                               size_t c_stride) {
    const svbool_t pg = svptrue_b16();

    // 256-bit SVE: 16 fp16 lanes
    const size_t K_STEP  = 16;
    const size_t K_STEP2 = 32;

    const size_t K_rounded2 = K & ~(K_STEP2 - 1);
    const size_t K_rounded1 = K & ~(K_STEP  - 1);

    size_t i = 0;

    // Main path: compute 8 output channels at once.
    // This reuses one A vector for 8 B rows.
    for (; i + 8 <= N; i += 8) {
        svfloat16_t acc0 = svdup_f16(0.0f);
        svfloat16_t acc1 = svdup_f16(0.0f);
        svfloat16_t acc2 = svdup_f16(0.0f);
        svfloat16_t acc3 = svdup_f16(0.0f);
        svfloat16_t acc4 = svdup_f16(0.0f);
        svfloat16_t acc5 = svdup_f16(0.0f);
        svfloat16_t acc6 = svdup_f16(0.0f);
        svfloat16_t acc7 = svdup_f16(0.0f);

        size_t k = 0;

        // Unroll K by 2 vectors to reduce loop overhead and give compiler more scheduling room.
        for (; k < K_rounded2; k += K_STEP2) {
            svfloat16_t a0 = svld1_f16(pg, (const __fp16 *) &A[k]);

            svfloat16_t b0 = svld1_f16(pg, (const __fp16 *) &B[(i + 0) * K + k]);
            svfloat16_t b1 = svld1_f16(pg, (const __fp16 *) &B[(i + 1) * K + k]);
            svfloat16_t b2 = svld1_f16(pg, (const __fp16 *) &B[(i + 2) * K + k]);
            svfloat16_t b3 = svld1_f16(pg, (const __fp16 *) &B[(i + 3) * K + k]);
            svfloat16_t b4 = svld1_f16(pg, (const __fp16 *) &B[(i + 4) * K + k]);
            svfloat16_t b5 = svld1_f16(pg, (const __fp16 *) &B[(i + 5) * K + k]);
            svfloat16_t b6 = svld1_f16(pg, (const __fp16 *) &B[(i + 6) * K + k]);
            svfloat16_t b7 = svld1_f16(pg, (const __fp16 *) &B[(i + 7) * K + k]);

            acc0 = svmla_f16_m(pg, acc0, a0, b0);
            acc1 = svmla_f16_m(pg, acc1, a0, b1);
            acc2 = svmla_f16_m(pg, acc2, a0, b2);
            acc3 = svmla_f16_m(pg, acc3, a0, b3);
            acc4 = svmla_f16_m(pg, acc4, a0, b4);
            acc5 = svmla_f16_m(pg, acc5, a0, b5);
            acc6 = svmla_f16_m(pg, acc6, a0, b6);
            acc7 = svmla_f16_m(pg, acc7, a0, b7);

            svfloat16_t a1 = svld1_f16(pg, (const __fp16 *) &A[k + K_STEP]);

            b0 = svld1_f16(pg, (const __fp16 *) &B[(i + 0) * K + k + K_STEP]);
            b1 = svld1_f16(pg, (const __fp16 *) &B[(i + 1) * K + k + K_STEP]);
            b2 = svld1_f16(pg, (const __fp16 *) &B[(i + 2) * K + k + K_STEP]);
            b3 = svld1_f16(pg, (const __fp16 *) &B[(i + 3) * K + k + K_STEP]);
            b4 = svld1_f16(pg, (const __fp16 *) &B[(i + 4) * K + k + K_STEP]);
            b5 = svld1_f16(pg, (const __fp16 *) &B[(i + 5) * K + k + K_STEP]);
            b6 = svld1_f16(pg, (const __fp16 *) &B[(i + 6) * K + k + K_STEP]);
            b7 = svld1_f16(pg, (const __fp16 *) &B[(i + 7) * K + k + K_STEP]);

            acc0 = svmla_f16_m(pg, acc0, a1, b0);
            acc1 = svmla_f16_m(pg, acc1, a1, b1);
            acc2 = svmla_f16_m(pg, acc2, a1, b2);
            acc3 = svmla_f16_m(pg, acc3, a1, b3);
            acc4 = svmla_f16_m(pg, acc4, a1, b4);
            acc5 = svmla_f16_m(pg, acc5, a1, b5);
            acc6 = svmla_f16_m(pg, acc6, a1, b6);
            acc7 = svmla_f16_m(pg, acc7, a1, b7);
        }

        // One remaining full SVE vector.
        if (k < K_rounded1) {
            svfloat16_t a0 = svld1_f16(pg, (const __fp16 *) &A[k]);

            svfloat16_t b0 = svld1_f16(pg, (const __fp16 *) &B[(i + 0) * K + k]);
            svfloat16_t b1 = svld1_f16(pg, (const __fp16 *) &B[(i + 1) * K + k]);
            svfloat16_t b2 = svld1_f16(pg, (const __fp16 *) &B[(i + 2) * K + k]);
            svfloat16_t b3 = svld1_f16(pg, (const __fp16 *) &B[(i + 3) * K + k]);
            svfloat16_t b4 = svld1_f16(pg, (const __fp16 *) &B[(i + 4) * K + k]);
            svfloat16_t b5 = svld1_f16(pg, (const __fp16 *) &B[(i + 5) * K + k]);
            svfloat16_t b6 = svld1_f16(pg, (const __fp16 *) &B[(i + 6) * K + k]);
            svfloat16_t b7 = svld1_f16(pg, (const __fp16 *) &B[(i + 7) * K + k]);

            acc0 = svmla_f16_m(pg, acc0, a0, b0);
            acc1 = svmla_f16_m(pg, acc1, a0, b1);
            acc2 = svmla_f16_m(pg, acc2, a0, b2);
            acc3 = svmla_f16_m(pg, acc3, a0, b3);
            acc4 = svmla_f16_m(pg, acc4, a0, b4);
            acc5 = svmla_f16_m(pg, acc5, a0, b5);
            acc6 = svmla_f16_m(pg, acc6, a0, b6);
            acc7 = svmla_f16_m(pg, acc7, a0, b7);

            k += K_STEP;
        }

        // K tail: 1..15 fp16
        if (k < K) {
            svbool_t pg_tail = svwhilelt_b16(k, K);

            svfloat16_t a0 = svld1_f16(pg_tail, (const __fp16 *) &A[k]);

            svfloat16_t b0 = svld1_f16(pg_tail, (const __fp16 *) &B[(i + 0) * K + k]);
            svfloat16_t b1 = svld1_f16(pg_tail, (const __fp16 *) &B[(i + 1) * K + k]);
            svfloat16_t b2 = svld1_f16(pg_tail, (const __fp16 *) &B[(i + 2) * K + k]);
            svfloat16_t b3 = svld1_f16(pg_tail, (const __fp16 *) &B[(i + 3) * K + k]);
            svfloat16_t b4 = svld1_f16(pg_tail, (const __fp16 *) &B[(i + 4) * K + k]);
            svfloat16_t b5 = svld1_f16(pg_tail, (const __fp16 *) &B[(i + 5) * K + k]);
            svfloat16_t b6 = svld1_f16(pg_tail, (const __fp16 *) &B[(i + 6) * K + k]);
            svfloat16_t b7 = svld1_f16(pg_tail, (const __fp16 *) &B[(i + 7) * K + k]);

            acc0 = svmla_f16_m(pg_tail, acc0, a0, b0);
            acc1 = svmla_f16_m(pg_tail, acc1, a0, b1);
            acc2 = svmla_f16_m(pg_tail, acc2, a0, b2);
            acc3 = svmla_f16_m(pg_tail, acc3, a0, b3);
            acc4 = svmla_f16_m(pg_tail, acc4, a0, b4);
            acc5 = svmla_f16_m(pg_tail, acc5, a0, b5);
            acc6 = svmla_f16_m(pg_tail, acc6, a0, b6);
            acc7 = svmla_f16_m(pg_tail, acc7, a0, b7);
        }

        C[(i + 0) * c_stride] = (float) svaddv_f16(pg, acc0);
        C[(i + 1) * c_stride] = (float) svaddv_f16(pg, acc1);
        C[(i + 2) * c_stride] = (float) svaddv_f16(pg, acc2);
        C[(i + 3) * c_stride] = (float) svaddv_f16(pg, acc3);
        C[(i + 4) * c_stride] = (float) svaddv_f16(pg, acc4);
        C[(i + 5) * c_stride] = (float) svaddv_f16(pg, acc5);
        C[(i + 6) * c_stride] = (float) svaddv_f16(pg, acc6);
        C[(i + 7) * c_stride] = (float) svaddv_f16(pg, acc7);
    }

    // N tail: 4 outputs at once.
    for (; i + 4 <= N; i += 4) {
        svfloat16_t acc0 = svdup_f16(0.0f);
        svfloat16_t acc1 = svdup_f16(0.0f);
        svfloat16_t acc2 = svdup_f16(0.0f);
        svfloat16_t acc3 = svdup_f16(0.0f);

        size_t k = 0;

        for (; k < K_rounded2; k += K_STEP2) {
            svfloat16_t a0 = svld1_f16(pg, (const __fp16 *) &A[k]);

            svfloat16_t b0 = svld1_f16(pg, (const __fp16 *) &B[(i + 0) * K + k]);
            svfloat16_t b1 = svld1_f16(pg, (const __fp16 *) &B[(i + 1) * K + k]);
            svfloat16_t b2 = svld1_f16(pg, (const __fp16 *) &B[(i + 2) * K + k]);
            svfloat16_t b3 = svld1_f16(pg, (const __fp16 *) &B[(i + 3) * K + k]);

            acc0 = svmla_f16_m(pg, acc0, a0, b0);
            acc1 = svmla_f16_m(pg, acc1, a0, b1);
            acc2 = svmla_f16_m(pg, acc2, a0, b2);
            acc3 = svmla_f16_m(pg, acc3, a0, b3);

            svfloat16_t a1 = svld1_f16(pg, (const __fp16 *) &A[k + K_STEP]);

            b0 = svld1_f16(pg, (const __fp16 *) &B[(i + 0) * K + k + K_STEP]);
            b1 = svld1_f16(pg, (const __fp16 *) &B[(i + 1) * K + k + K_STEP]);
            b2 = svld1_f16(pg, (const __fp16 *) &B[(i + 2) * K + k + K_STEP]);
            b3 = svld1_f16(pg, (const __fp16 *) &B[(i + 3) * K + k + K_STEP]);

            acc0 = svmla_f16_m(pg, acc0, a1, b0);
            acc1 = svmla_f16_m(pg, acc1, a1, b1);
            acc2 = svmla_f16_m(pg, acc2, a1, b2);
            acc3 = svmla_f16_m(pg, acc3, a1, b3);
        }

        if (k < K_rounded1) {
            svfloat16_t a0 = svld1_f16(pg, (const __fp16 *) &A[k]);

            svfloat16_t b0 = svld1_f16(pg, (const __fp16 *) &B[(i + 0) * K + k]);
            svfloat16_t b1 = svld1_f16(pg, (const __fp16 *) &B[(i + 1) * K + k]);
            svfloat16_t b2 = svld1_f16(pg, (const __fp16 *) &B[(i + 2) * K + k]);
            svfloat16_t b3 = svld1_f16(pg, (const __fp16 *) &B[(i + 3) * K + k]);

            acc0 = svmla_f16_m(pg, acc0, a0, b0);
            acc1 = svmla_f16_m(pg, acc1, a0, b1);
            acc2 = svmla_f16_m(pg, acc2, a0, b2);
            acc3 = svmla_f16_m(pg, acc3, a0, b3);

            k += K_STEP;
        }

        if (k < K) {
            svbool_t pg_tail = svwhilelt_b16(k, K);

            svfloat16_t a0 = svld1_f16(pg_tail, (const __fp16 *) &A[k]);

            svfloat16_t b0 = svld1_f16(pg_tail, (const __fp16 *) &B[(i + 0) * K + k]);
            svfloat16_t b1 = svld1_f16(pg_tail, (const __fp16 *) &B[(i + 1) * K + k]);
            svfloat16_t b2 = svld1_f16(pg_tail, (const __fp16 *) &B[(i + 2) * K + k]);
            svfloat16_t b3 = svld1_f16(pg_tail, (const __fp16 *) &B[(i + 3) * K + k]);

            acc0 = svmla_f16_m(pg_tail, acc0, a0, b0);
            acc1 = svmla_f16_m(pg_tail, acc1, a0, b1);
            acc2 = svmla_f16_m(pg_tail, acc2, a0, b2);
            acc3 = svmla_f16_m(pg_tail, acc3, a0, b3);
        }

        C[(i + 0) * c_stride] = (float) svaddv_f16(pg, acc0);
        C[(i + 1) * c_stride] = (float) svaddv_f16(pg, acc1);
        C[(i + 2) * c_stride] = (float) svaddv_f16(pg, acc2);
        C[(i + 3) * c_stride] = (float) svaddv_f16(pg, acc3);
    }

    // Final N tail: 1 output at once, still SVE, no scalar fallback.
    for (; i < N; i++) {
        svfloat16_t acc = svdup_f16(0.0f);

        size_t k = 0;

        for (; k < K_rounded2; k += K_STEP2) {
            svfloat16_t a0 = svld1_f16(pg, (const __fp16 *) &A[k]);
            svfloat16_t b0 = svld1_f16(pg, (const __fp16 *) &B[i * K + k]);
            acc = svmla_f16_m(pg, acc, a0, b0);

            svfloat16_t a1 = svld1_f16(pg, (const __fp16 *) &A[k + K_STEP]);
            svfloat16_t b1 = svld1_f16(pg, (const __fp16 *) &B[i * K + k + K_STEP]);
            acc = svmla_f16_m(pg, acc, a1, b1);
        }

        if (k < K_rounded1) {
            svfloat16_t a0 = svld1_f16(pg, (const __fp16 *) &A[k]);
            svfloat16_t b0 = svld1_f16(pg, (const __fp16 *) &B[i * K + k]);
            acc = svmla_f16_m(pg, acc, a0, b0);
            k += K_STEP;
        }

        if (k < K) {
            svbool_t pg_tail = svwhilelt_b16(k, K);
            svfloat16_t a0 = svld1_f16(pg_tail, (const __fp16 *) &A[k]);
            svfloat16_t b0 = svld1_f16(pg_tail, (const __fp16 *) &B[i * K + k]);
            acc = svmla_f16_m(pg_tail, acc, a0, b0);
        }

        C[i * c_stride] = (float) svaddv_f16(pg, acc);
    }
}


void ggml_matmul_f16_4x4_kernel(size_t na, size_t nb, int dim,
                   const void* GGML_RESTRICT a, const void* GGML_RESTRICT b,
                   void* GGML_RESTRICT c, size_t c_column_bytes) {
    const size_t M = na;
    const size_t N = nb;
    const size_t K = dim;
    ggml_fp16_t* GGML_RESTRICT A = (ggml_fp16_t*)(a);
    ggml_fp16_t* GGML_RESTRICT B = (ggml_fp16_t*)(b);
    float32_t* GGML_RESTRICT C = (float32_t*)(c);
    size_t c_stride = c_column_bytes / sizeof(float32_t);

    const size_t K_rounded = K & ~((1 << 3) - 1);  // 处理K不是8的倍数的情况

    // Fast path for N=1 (decode, M output channels × 1 token).
    // Each K_STEP=8: load 1 B vector, inner loop over M rows with f16 fma.
    // Reduces B loads by (M-1)/M compared to N_remainder path (1 SVE dot per row).
    if (N == 1) {
        for (size_t j = 0; j < M; j++) {
            float16x8_t acc = vdupq_n_f16(0);

            size_t k = 0;
            for (; k < K_rounded; k += 8) {
                float16x8_t b_v = vld1q_f16((const __fp16 *) &B[k]);
                float16x8_t a_v = vld1q_f16((const __fp16 *) &A[j * K + k]);
                acc = vfmaq_f16(acc, a_v, b_v);
            }

            if (k < K) {
                svbool_t pg = svwhilelt_b16(k, K);
                float16x8_t b_v = svget_neonq_f16(
                    svld1_f16(pg, (const __fp16 *) &B[k]));
                float16x8_t a_v = svget_neonq_f16(
                    svld1_f16(pg, (const __fp16 *) &A[j * K + k]));
                acc = vfmaq_f16(acc, a_v, b_v);
            }

            float32_t sum =
                vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc))) +
                vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc)));

            C[j] = sum;
        }

        return;
    }

    size_t N_rounded = N & ~((1 << 2) - 1);
    size_t M_rounded = M & ~((1 << 2) - 1);

    for (size_t i = 0; i < N_rounded; i += 4) {
        for (size_t j = 0; j < M_rounded; j += 4) {
            float16x8_t acc00 = vdupq_n_f16(0);
            float16x8_t acc01 = vdupq_n_f16(0);
            float16x8_t acc02 = vdupq_n_f16(0);
            float16x8_t acc03 = vdupq_n_f16(0);
            float16x8_t acc10 = vdupq_n_f16(0);
            float16x8_t acc11 = vdupq_n_f16(0);
            float16x8_t acc12 = vdupq_n_f16(0);
            float16x8_t acc13 = vdupq_n_f16(0);
            float16x8_t acc20 = vdupq_n_f16(0);
            float16x8_t acc21 = vdupq_n_f16(0);
            float16x8_t acc22 = vdupq_n_f16(0);
            float16x8_t acc23 = vdupq_n_f16(0);
            float16x8_t acc30 = vdupq_n_f16(0);
            float16x8_t acc31 = vdupq_n_f16(0);
            float16x8_t acc32 = vdupq_n_f16(0);
            float16x8_t acc33 = vdupq_n_f16(0);

            for (size_t k = 0; k < K_rounded; k += 8) {
                float16x8_t j0 = vld1q_f16((const __fp16 *)(&A[(j + 0) * K + k]));
                float16x8_t j1 = vld1q_f16((const __fp16 *)(&A[(j + 1) * K + k]));
                float16x8_t j2 = vld1q_f16((const __fp16 *)(&A[(j + 2) * K + k]));
                float16x8_t j3 = vld1q_f16((const __fp16 *)(&A[(j + 3) * K + k]));
                float16x8_t i0 = vld1q_f16((const __fp16 *)(&B[(i + 0) * K + k]));
                float16x8_t i1 = vld1q_f16((const __fp16 *)(&B[(i + 1) * K + k]));
                float16x8_t i2 = vld1q_f16((const __fp16 *)(&B[(i + 2) * K + k]));
                float16x8_t i3 = vld1q_f16((const __fp16 *)(&B[(i + 3) * K + k]));

                acc00 = vfmaq_f16(acc00, i0, j0);
                acc01 = vfmaq_f16(acc01, i0, j1);
                acc02 = vfmaq_f16(acc02, i0, j2);
                acc03 = vfmaq_f16(acc03, i0, j3);
                acc10 = vfmaq_f16(acc10, i1, j0);
                acc11 = vfmaq_f16(acc11, i1, j1);
                acc12 = vfmaq_f16(acc12, i1, j2);
                acc13 = vfmaq_f16(acc13, i1, j3);
                acc20 = vfmaq_f16(acc20, i2, j0);
                acc21 = vfmaq_f16(acc21, i2, j1);
                acc22 = vfmaq_f16(acc22, i2, j2);
                acc23 = vfmaq_f16(acc23, i2, j3);
                acc30 = vfmaq_f16(acc30, i3, j0);
                acc31 = vfmaq_f16(acc31, i3, j1);
                acc32 = vfmaq_f16(acc32, i3, j2);
                acc33 = vfmaq_f16(acc33, i3, j3);
            }

            // 处理剩余的K元素(1-7个)
            if (K_rounded < K) {
                svbool_t pg = svwhilelt_b16(K_rounded, K);
                float16x8_t j0 = svget_neonq_f16(svld1_f16(pg, (const __fp16 *)(&A[(j + 0) * K + K_rounded])));
                float16x8_t j1 = svget_neonq_f16(svld1_f16(pg, (const __fp16 *)(&A[(j + 1) * K + K_rounded])));
                float16x8_t j2 = svget_neonq_f16(svld1_f16(pg, (const __fp16 *)(&A[(j + 2) * K + K_rounded])));
                float16x8_t j3 = svget_neonq_f16(svld1_f16(pg, (const __fp16 *)(&A[(j + 3) * K + K_rounded])));
                float16x8_t i0 = svget_neonq_f16(svld1_f16(pg, (const __fp16 *)(&B[(i + 0) * K + K_rounded])));
                float16x8_t i1 = svget_neonq_f16(svld1_f16(pg, (const __fp16 *)(&B[(i + 1) * K + K_rounded])));
                float16x8_t i2 = svget_neonq_f16(svld1_f16(pg, (const __fp16 *)(&B[(i + 2) * K + K_rounded])));
                float16x8_t i3 = svget_neonq_f16(svld1_f16(pg, (const __fp16 *)(&B[(i + 3) * K + K_rounded])));

                acc00 = vfmaq_f16(acc00, i0, j0);
                acc01 = vfmaq_f16(acc01, i0, j1);
                acc02 = vfmaq_f16(acc02, i0, j2);
                acc03 = vfmaq_f16(acc03, i0, j3);
                acc10 = vfmaq_f16(acc10, i1, j0);
                acc11 = vfmaq_f16(acc11, i1, j1);
                acc12 = vfmaq_f16(acc12, i1, j2);
                acc13 = vfmaq_f16(acc13, i1, j3);
                acc20 = vfmaq_f16(acc20, i2, j0);
                acc21 = vfmaq_f16(acc21, i2, j1);
                acc22 = vfmaq_f16(acc22, i2, j2);
                acc23 = vfmaq_f16(acc23, i2, j3);
                acc30 = vfmaq_f16(acc30, i3, j0);
                acc31 = vfmaq_f16(acc31, i3, j1);
                acc32 = vfmaq_f16(acc32, i3, j2);
                acc33 = vfmaq_f16(acc33, i3, j3);
            }

            // Horizontal reduction
            float32_t r00 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc00))) +
                           vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc00)));
            float32_t r01 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc01))) +
                           vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc01)));
            float32_t r02 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc02))) +
                           vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc02)));
            float32_t r03 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc03))) +
                           vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc03)));
            float32_t r10 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc10))) +
                           vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc10)));
            float32_t r11 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc11))) +
                           vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc11)));
            float32_t r12 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc12))) +
                           vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc12)));
            float32_t r13 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc13))) +
                           vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc13)));
            float32_t r20 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc20))) +
                           vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc20)));
            float32_t r21 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc21))) +
                           vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc21)));
            float32_t r22 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc22))) +
                           vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc22)));
            float32_t r23 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc23))) +
                           vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc23)));
            float32_t r30 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc30))) +
                           vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc30)));
            float32_t r31 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc31))) +
                           vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc31)));
            float32_t r32 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc32))) +
                           vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc32)));
            float32_t r33 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc33))) +
                           vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc33)));

            C[((i + 0) * c_stride) + (j + 0)] = r00;
            C[((i + 0) * c_stride) + (j + 1)] = r01;
            C[((i + 0) * c_stride) + (j + 2)] = r02;
            C[((i + 0) * c_stride) + (j + 3)] = r03;
            C[((i + 1) * c_stride) + (j + 0)] = r10;
            C[((i + 1) * c_stride) + (j + 1)] = r11;
            C[((i + 1) * c_stride) + (j + 2)] = r12;
            C[((i + 1) * c_stride) + (j + 3)] = r13;
            C[((i + 2) * c_stride) + (j + 0)] = r20;
            C[((i + 2) * c_stride) + (j + 1)] = r21;
            C[((i + 2) * c_stride) + (j + 2)] = r22;
            C[((i + 2) * c_stride) + (j + 3)] = r23;
            C[((i + 3) * c_stride) + (j + 0)] = r30;
            C[((i + 3) * c_stride) + (j + 1)] = r31;
            C[((i + 3) * c_stride) + (j + 2)] = r32;
            C[((i + 3) * c_stride) + (j + 3)] = r33;
        }
    }

    // Handle leftovers - M remainder (1-3 rows)
    // Use batched NEON to avoid vec_dot_f16 overhead
    size_t M_rem = M - M_rounded;
    if (M_rem > 0) {
        const size_t K_rounded = K & ~((1 << 3) - 1);
        // Process N in batches
        size_t i = 0;
        for (; i + 4 <= N_rounded; i += 4) {
            // Process M_rem rows x 4 B-rows
            for (size_t j = M_rounded; j < M; j++) {
                float16x8_t acc0 = vdupq_n_f16(0);
                float16x8_t acc1 = vdupq_n_f16(0);
                float16x8_t acc2 = vdupq_n_f16(0);
                float16x8_t acc3 = vdupq_n_f16(0);
                for (size_t k = 0; k < K_rounded; k += 8) {
                    float16x8_t a_v = vld1q_f16((const __fp16 *)(&A[j * K + k]));
                    float16x8_t b0 = vld1q_f16((const __fp16 *)(&B[(i + 0) * K + k]));
                    float16x8_t b1 = vld1q_f16((const __fp16 *)(&B[(i + 1) * K + k]));
                    float16x8_t b2 = vld1q_f16((const __fp16 *)(&B[(i + 2) * K + k]));
                    float16x8_t b3 = vld1q_f16((const __fp16 *)(&B[(i + 3) * K + k]));
                    acc0 = vfmaq_f16(acc0, a_v, b0);
                    acc1 = vfmaq_f16(acc1, a_v, b1);
                    acc2 = vfmaq_f16(acc2, a_v, b2);
                    acc3 = vfmaq_f16(acc3, a_v, b3);
                }
                if (K_rounded < K) {
                    svbool_t pg = svwhilelt_b16(K_rounded, K);
                    float16x8_t a_v = svget_neonq_f16(svld1_f16(pg, (const __fp16 *)(&A[j * K + K_rounded])));
                    float16x8_t b0 = svget_neonq_f16(svld1_f16(pg, (const __fp16 *)(&B[(i + 0) * K + K_rounded])));
                    float16x8_t b1 = svget_neonq_f16(svld1_f16(pg, (const __fp16 *)(&B[(i + 1) * K + K_rounded])));
                    float16x8_t b2 = svget_neonq_f16(svld1_f16(pg, (const __fp16 *)(&B[(i + 2) * K + K_rounded])));
                    float16x8_t b3 = svget_neonq_f16(svld1_f16(pg, (const __fp16 *)(&B[(i + 3) * K + K_rounded])));
                    acc0 = vfmaq_f16(acc0, a_v, b0);
                    acc1 = vfmaq_f16(acc1, a_v, b1);
                    acc2 = vfmaq_f16(acc2, a_v, b2);
                    acc3 = vfmaq_f16(acc3, a_v, b3);
                }
                float32_t r0 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc0))) +
                               vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc0)));
                float32_t r1 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc1))) +
                               vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc1)));
                float32_t r2 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc2))) +
                               vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc2)));
                float32_t r3 = vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc3))) +
                               vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc3)));
                C[(i + 0) * c_stride + j] = r0;
                C[(i + 1) * c_stride + j] = r1;
                C[(i + 2) * c_stride + j] = r2;
                C[(i + 3) * c_stride + j] = r3;
            }
        }
        // Remaining N columns (0-3), handle each M_rem row individually
        for (; i < N_rounded; i++) {
            for (size_t j = M_rounded; j < M; j++) {
                float16x8_t acc = vdupq_n_f16(0);
                for (size_t k = 0; k < K_rounded; k += 8) {
                    float16x8_t a_v = vld1q_f16((const __fp16 *)(&A[j * K + k]));
                    float16x8_t b_v = vld1q_f16((const __fp16 *)(&B[i * K + k]));
                    acc = vfmaq_f16(acc, a_v, b_v);
                }
                if (K_rounded < K) {
                    svbool_t pg = svwhilelt_b16(K_rounded, K);
                    float16x8_t a_v = svget_neonq_f16(svld1_f16(pg, (const __fp16 *)(&A[j * K + K_rounded])));
                    float16x8_t b_v = svget_neonq_f16(svld1_f16(pg, (const __fp16 *)(&B[i * K + K_rounded])));
                    acc = vfmaq_f16(acc, a_v, b_v);
                }
                float32_t r = vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc))) +
                              vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc)));
                C[i * c_stride + j] = r;
            }
        }
    }
    // Handle N remainder (1-3 columns), for already-processed M rows
    // Use SVE256 for the M-loop to process 16 fp16 per K step (vs NEON's 8)
    for (size_t i = N_rounded; i < N; i++) {
        // NEON fallback for non-SVE targets
        for (size_t j = 0; j < M; j++) {
            float16x8_t acc = vdupq_n_f16(0);
            for (size_t k = 0; k < K_rounded; k += 8) {
                float16x8_t a_v = vld1q_f16((const __fp16 *)(&A[j * K + k]));
                float16x8_t b_v = vld1q_f16((const __fp16 *)(&B[i * K + k]));
                acc = vfmaq_f16(acc, a_v, b_v);
            }
            if (K_rounded < K) {
                svbool_t pg = svwhilelt_b16(K_rounded, K);
                float16x8_t a_v = svget_neonq_f16(svld1_f16(pg, (const __fp16 *)(&A[j * K + K_rounded])));
                float16x8_t b_v = svget_neonq_f16(svld1_f16(pg, (const __fp16 *)(&B[i * K + K_rounded])));
                acc = vfmaq_f16(acc, a_v, b_v);
            }
            float32_t r = vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc))) +
                          vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc)));
            C[i * c_stride + j] = r;
        }
    }
}

void
matmul_outer_8x16_b_micro_packA_f32(size_t M,
                                    size_t N,
                                    size_t K,
                                    const void* a,
                                    size_t a_column_bytes,
                                    const void* b,
                                    size_t b_column_bytes,
                                    void* GGML_RESTRICT c,
                                    size_t c_column_bytes)
{
    assert(M % 16 == 0 && N % 8 == 0);
    svbool_t pg_all = svptrue_b32();
    float32_t* GGML_RESTRICT C = (float32_t*)(c);
    const float32_t* GGML_RESTRICT B = (const float32_t*)(b);
    size_t c_stride = c_column_bytes / sizeof(float32_t);
    size_t b_stride = b_column_bytes / sizeof(float32_t);
    size_t a_stride = a_column_bytes / sizeof(float32_t);
    const float32_t* pa = NULL;
    pa = (const float32_t*)(((const matmul_outer_pack_t*)a)->p);
    for (size_t i = 0; i < N; i += 8) {
        for (size_t j = 0; j < M; j += 16) {
            svfloat32_t acc00;
            svfloat32_t acc01;
            svfloat32_t acc02;
            svfloat32_t acc03;
            svfloat32_t acc04;
            svfloat32_t acc05;
            svfloat32_t acc06;
            svfloat32_t acc07;
            svfloat32_t acc10;
            svfloat32_t acc11;
            svfloat32_t acc12;
            svfloat32_t acc13;
            svfloat32_t acc14;
            svfloat32_t acc15;
            svfloat32_t acc16;
            svfloat32_t acc17;
#define LOAD_C(a, b)                                                                                                   \
    do {                                                                                                               \
        (acc##a##b) = svld1_f32(pg_all, &C[((i + (b)) * c_stride) + (j + 8 * a)]);                                     \
    } while (0)
            LOAD_C(0, 0);
            LOAD_C(0, 1);
            LOAD_C(0, 2);
            LOAD_C(0, 3);
            LOAD_C(0, 4);
            LOAD_C(0, 5);
            LOAD_C(0, 6);
            LOAD_C(0, 7);
            LOAD_C(1, 0);
            LOAD_C(1, 1);
            LOAD_C(1, 2);
            LOAD_C(1, 3);
            LOAD_C(1, 4);
            LOAD_C(1, 5);
            LOAD_C(1, 6);
            LOAD_C(1, 7);
#undef LOAD_C
            const float32_t* pa0 = &((const float32_t*)pa)[(j + 0 * 8) * a_stride];
            const float32_t* pa1 = &((const float32_t*)pa)[(j + 1 * 8) * a_stride];
            const float32_t* pb0 = &B[(i + 0) * b_stride];
            const float32_t* pb1 = &B[(i + 1) * b_stride];
            const float32_t* pb2 = &B[(i + 2) * b_stride];
            const float32_t* pb3 = &B[(i + 3) * b_stride];
            const float32_t* pb4 = &B[(i + 4) * b_stride];
            const float32_t* pb5 = &B[(i + 5) * b_stride];
            const float32_t* pb6 = &B[(i + 6) * b_stride];
            const float32_t* pb7 = &B[(i + 7) * b_stride];
            for (size_t k = 0; k < K; k += 4) {
                svbool_t pg = k + 4 <= K ? pg_all : svwhilelt_b32(k, K);
                svfloat32_t b0 = svld1rq_f32(pg, &pb0[k]);
                svfloat32_t b1 = svld1rq_f32(pg, &pb1[k]);
                svfloat32_t b2 = svld1rq_f32(pg, &pb2[k]);
                svfloat32_t b3 = svld1rq_f32(pg, &pb3[k]);
                svfloat32_t b4 = svld1rq_f32(pg, &pb4[k]);
                svfloat32_t b5 = svld1rq_f32(pg, &pb5[k]);
                svfloat32_t b6 = svld1rq_f32(pg, &pb6[k]);
                svfloat32_t b7 = svld1rq_f32(pg, &pb7[k]);
                svfloat32_t a0M1;
                svfloat32_t a1M1;
                svfloat32_t a0M2;
                svfloat32_t a1M2;
#define KERNEL8x2_IDX_I                                                                                                \
    do {                                                                                                               \
        a0M1 = svld1_f32(pg_all, &pa0[(k + 0) * 8]);                                                                   \
        a1M1 = svld1_f32(pg_all, &pa1[(k + 0) * 8]);                                                                   \
        a0M2 = svld1_f32(pg_all, &pa0[(k + 1) * 8]);                                                                   \
        a1M2 = svld1_f32(pg_all, &pa1[(k + 1) * 8]);                                                                   \
        acc00 = svmla_lane_f32(acc00, a0M1, b0, 0);                                                                    \
        acc01 = svmla_lane_f32(acc01, a0M1, b1, 0);                                                                    \
        acc02 = svmla_lane_f32(acc02, a0M1, b2, 0);                                                                    \
        acc03 = svmla_lane_f32(acc03, a0M1, b3, 0);                                                                    \
        acc04 = svmla_lane_f32(acc04, a0M1, b4, 0);                                                                    \
        acc05 = svmla_lane_f32(acc05, a0M1, b5, 0);                                                                    \
        acc06 = svmla_lane_f32(acc06, a0M1, b6, 0);                                                                    \
        acc07 = svmla_lane_f32(acc07, a0M1, b7, 0);                                                                    \
        acc10 = svmla_lane_f32(acc10, a1M1, b0, 0);                                                                    \
        acc11 = svmla_lane_f32(acc11, a1M1, b1, 0);                                                                    \
        acc12 = svmla_lane_f32(acc12, a1M1, b2, 0);                                                                    \
        acc13 = svmla_lane_f32(acc13, a1M1, b3, 0);                                                                    \
        acc14 = svmla_lane_f32(acc14, a1M1, b4, 0);                                                                    \
        acc15 = svmla_lane_f32(acc15, a1M1, b5, 0);                                                                    \
        acc16 = svmla_lane_f32(acc16, a1M1, b6, 0);                                                                    \
        acc17 = svmla_lane_f32(acc17, a1M1, b7, 0);                                                                    \
    } while (0)
#define KERNEL8x2_IDX(idx, ml, mc)                                                                                     \
    do {                                                                                                               \
        a0M##ml = svld1_f32(pg_all, &pa0[(k + (idx + 1)) * 8]);                                                        \
        acc00 = svmla_lane_f32(acc00, a0M##mc, b0, (idx));                                                             \
        acc01 = svmla_lane_f32(acc01, a0M##mc, b1, (idx));                                                             \
        acc02 = svmla_lane_f32(acc02, a0M##mc, b2, (idx));                                                             \
        acc03 = svmla_lane_f32(acc03, a0M##mc, b3, (idx));                                                             \
        acc04 = svmla_lane_f32(acc04, a0M##mc, b4, (idx));                                                             \
        acc05 = svmla_lane_f32(acc05, a0M##mc, b5, (idx));                                                             \
        acc06 = svmla_lane_f32(acc06, a0M##mc, b6, (idx));                                                             \
        acc07 = svmla_lane_f32(acc07, a0M##mc, b7, (idx));                                                             \
        a1M##ml = svld1_f32(pg_all, &pa1[(k + (idx + 1)) * 8]);                                                        \
        acc10 = svmla_lane_f32(acc10, a1M##mc, b0, (idx));                                                             \
        acc11 = svmla_lane_f32(acc11, a1M##mc, b1, (idx));                                                             \
        acc12 = svmla_lane_f32(acc12, a1M##mc, b2, (idx));                                                             \
        acc13 = svmla_lane_f32(acc13, a1M##mc, b3, (idx));                                                             \
        acc14 = svmla_lane_f32(acc14, a1M##mc, b4, (idx));                                                             \
        acc15 = svmla_lane_f32(acc15, a1M##mc, b5, (idx));                                                             \
        acc16 = svmla_lane_f32(acc16, a1M##mc, b6, (idx));                                                             \
        acc17 = svmla_lane_f32(acc17, a1M##mc, b7, (idx));                                                             \
    } while (0)
#define KERNEL8x2_IDX_E                                                                                                \
    do {                                                                                                               \
        acc00 = svmla_lane_f32(acc00, a0M2, b0, 3);                                                                    \
        acc01 = svmla_lane_f32(acc01, a0M2, b1, 3);                                                                    \
        acc02 = svmla_lane_f32(acc02, a0M2, b2, 3);                                                                    \
        acc03 = svmla_lane_f32(acc03, a0M2, b3, 3);                                                                    \
        acc04 = svmla_lane_f32(acc04, a0M2, b4, 3);                                                                    \
        acc05 = svmla_lane_f32(acc05, a0M2, b5, 3);                                                                    \
        acc06 = svmla_lane_f32(acc06, a0M2, b6, 3);                                                                    \
        acc07 = svmla_lane_f32(acc07, a0M2, b7, 3);                                                                    \
        acc10 = svmla_lane_f32(acc10, a1M2, b0, 3);                                                                    \
        acc11 = svmla_lane_f32(acc11, a1M2, b1, 3);                                                                    \
        acc12 = svmla_lane_f32(acc12, a1M2, b2, 3);                                                                    \
        acc13 = svmla_lane_f32(acc13, a1M2, b3, 3);                                                                    \
        acc14 = svmla_lane_f32(acc14, a1M2, b4, 3);                                                                    \
        acc15 = svmla_lane_f32(acc15, a1M2, b5, 3);                                                                    \
        acc16 = svmla_lane_f32(acc16, a1M2, b6, 3);                                                                    \
        acc17 = svmla_lane_f32(acc17, a1M2, b7, 3);                                                                    \
    } while (0)
                KERNEL8x2_IDX_I;
                KERNEL8x2_IDX(1, 1, 2);
                KERNEL8x2_IDX(2, 2, 1);
                KERNEL8x2_IDX_E;
#undef KERNEL8x2_IDX_I
#undef KERNEL8x2_IDX
#undef KERNEL8x2_IDX_E
            }
#define SAVE_C(a, b)                                                                                                   \
    do {                                                                                                               \
        svst1_f32(pg_all, &C[((i + b) * c_stride) + (j + 8 * a)], (acc##a##b));                                        \
    } while (0)
            SAVE_C(0, 0);
            SAVE_C(0, 1);
            SAVE_C(0, 2);
            SAVE_C(0, 3);
            SAVE_C(0, 4);
            SAVE_C(0, 5);
            SAVE_C(0, 6);
            SAVE_C(0, 7);
            SAVE_C(1, 0);
            SAVE_C(1, 1);
            SAVE_C(1, 2);
            SAVE_C(1, 3);
            SAVE_C(1, 4);
            SAVE_C(1, 5);
            SAVE_C(1, 6);
            SAVE_C(1, 7);
#undef SAVE_C
        }
    }
}
// Kunpeng 950 experimental NEON 16x4 kernel.
//
// Drop-in replacement for:
//   matmul_outer_8x16_b_micro_packA_f16()
//
// External interface is intentionally unchanged.
// The caller may still pass N in multiples of 8; this kernel processes N in
// internal tiles of 4, so no outer dispatch change is required for A/B testing.
//
// Tile:
//   M = 16
//   N = 4
//
// Register layout:
//   v16-v19 : B row 0..3, M[0..7]
//   v20-v23 : B row 0..3, M[8..15]
//
// Compared with the existing 16x8 kernel:
//   accumulators: 16 -> 8
//   live B regs : 8  -> 4
//
// This leaves substantially more register/scheduler headroom for Kunpeng 950.

void
matmul_outer_4x16_b_micro_packA_f16(size_t M,
                                    size_t N,
                                    size_t K,
                                    const void* a,
                                    size_t a_column_bytes,
                                    const void* b,
                                    size_t b_column_bytes,
                                    void* GGML_RESTRICT c,
                                    size_t c_column_bytes)
{
    GGML_UNUSED(a_column_bytes);

    const matmul_outer_pack_t* pack_a =
        (const matmul_outer_pack_t*) a;

    assert(M % 16 == 0 && N % 4 == 0);

    float32_t* GGML_RESTRICT C = (float32_t*) c;
    const float16_t* GGML_RESTRICT B = (const float16_t*) b;

    const size_t c_stride = c_column_bytes / sizeof(float32_t);
    const size_t b_stride = b_column_bytes / sizeof(float16_t);
    const size_t a_stride = pack_a->p_stride / sizeof(float16_t);

    const size_t a_stride_bytes = a_stride * sizeof(float16_t);
    const size_t b_stride_bytes = b_stride * sizeof(float16_t);
    const size_t c_stride_bytes = c_stride * sizeof(float32_t);

    asm volatile(
        "mov x1, #0\n"                         // i = 0
        "mov x10, %[B]\n"                      // pb0

        // ================================================================
        // N loop, 4 rows per tile
        // ================================================================
        "1:\n"
        "add x11, x10, %[b_stride]\n"
        "add x12, x11, %[b_stride]\n"
        "add x13, x12, %[b_stride]\n"

        "mov x2, #0\n"                         // j = 0
        "mov x3, %[pa]\n"                      // pa0 base

        // ================================================================
        // M loop, 16 cols
        // ================================================================
        "2:\n"
        "add x4, x3, %[a_stride_lsl3]\n"       // pa1 base

        "movi v16.16b, #0\n"
        "movi v17.16b, #0\n"
        "movi v18.16b, #0\n"
        "movi v19.16b, #0\n"
        "movi v20.16b, #0\n"
        "movi v21.16b, #0\n"
        "movi v22.16b, #0\n"
        "movi v23.16b, #0\n"

        // Incrementing packed-A pointers.
        "mov x8, x3\n"                         // A low-half ptr
        "mov x9, x4\n"                         // A high-half ptr
        "mov x7, #0\n"                         // B byte offset

        // Number of complete 8-K blocks (16 bytes each).
        "lsr x14, %[K2], #4\n"
        "cbz x14, 6f\n"

        // Preload first A pair.
        "ldr q8, [x8]\n"
        "ldr q9, [x9]\n"

        // ================================================================
        // Main loop: 8 FP16 K values / iteration
        // ================================================================
        "3:\n"
        // Load B vectors for this 8-K block.
        "ldr q0, [x10, x7]\n"
        "ldr q1, [x11, x7]\n"
        "ldr q2, [x12, x7]\n"
        "ldr q3, [x13, x7]\n"

        // lane 0, current A in v8/v9
        "fmla v16.8h, v8.8h, v0.h[0]\n"
        "fmla v17.8h, v8.8h, v1.h[0]\n"
        "fmla v18.8h, v8.8h, v2.h[0]\n"
        "fmla v19.8h, v8.8h, v3.h[0]\n"
        "fmla v20.8h, v9.8h, v0.h[0]\n"
        "fmla v21.8h, v9.8h, v1.h[0]\n"
        "fmla v22.8h, v9.8h, v2.h[0]\n"
        "fmla v23.8h, v9.8h, v3.h[0]\n"

        // Preload lane 1 A while lane 0 FMLAs retire.
        "ldr q10, [x8, #16]\n"
        "ldr q11, [x9, #16]\n"

        // lane 1
        "fmla v16.8h, v10.8h, v0.h[1]\n"
        "fmla v17.8h, v10.8h, v1.h[1]\n"
        "fmla v18.8h, v10.8h, v2.h[1]\n"
        "fmla v19.8h, v10.8h, v3.h[1]\n"
        "fmla v20.8h, v11.8h, v0.h[1]\n"
        "fmla v21.8h, v11.8h, v1.h[1]\n"
        "fmla v22.8h, v11.8h, v2.h[1]\n"
        "fmla v23.8h, v11.8h, v3.h[1]\n"

        // lane 2
        "ldr q8, [x8, #32]\n"
        "ldr q9, [x9, #32]\n"
        "fmla v16.8h, v8.8h, v0.h[2]\n"
        "fmla v17.8h, v8.8h, v1.h[2]\n"
        "fmla v18.8h, v8.8h, v2.h[2]\n"
        "fmla v19.8h, v8.8h, v3.h[2]\n"
        "fmla v20.8h, v9.8h, v0.h[2]\n"
        "fmla v21.8h, v9.8h, v1.h[2]\n"
        "fmla v22.8h, v9.8h, v2.h[2]\n"
        "fmla v23.8h, v9.8h, v3.h[2]\n"

        // lane 3
        "ldr q10, [x8, #48]\n"
        "ldr q11, [x9, #48]\n"
        "fmla v16.8h, v10.8h, v0.h[3]\n"
        "fmla v17.8h, v10.8h, v1.h[3]\n"
        "fmla v18.8h, v10.8h, v2.h[3]\n"
        "fmla v19.8h, v10.8h, v3.h[3]\n"
        "fmla v20.8h, v11.8h, v0.h[3]\n"
        "fmla v21.8h, v11.8h, v1.h[3]\n"
        "fmla v22.8h, v11.8h, v2.h[3]\n"
        "fmla v23.8h, v11.8h, v3.h[3]\n"

        // lane 4
        "ldr q8, [x8, #64]\n"
        "ldr q9, [x9, #64]\n"
        "fmla v16.8h, v8.8h, v0.h[4]\n"
        "fmla v17.8h, v8.8h, v1.h[4]\n"
        "fmla v18.8h, v8.8h, v2.h[4]\n"
        "fmla v19.8h, v8.8h, v3.h[4]\n"
        "fmla v20.8h, v9.8h, v0.h[4]\n"
        "fmla v21.8h, v9.8h, v1.h[4]\n"
        "fmla v22.8h, v9.8h, v2.h[4]\n"
        "fmla v23.8h, v9.8h, v3.h[4]\n"

        // lane 5
        "ldr q10, [x8, #80]\n"
        "ldr q11, [x9, #80]\n"
        "fmla v16.8h, v10.8h, v0.h[5]\n"
        "fmla v17.8h, v10.8h, v1.h[5]\n"
        "fmla v18.8h, v10.8h, v2.h[5]\n"
        "fmla v19.8h, v10.8h, v3.h[5]\n"
        "fmla v20.8h, v11.8h, v0.h[5]\n"
        "fmla v21.8h, v11.8h, v1.h[5]\n"
        "fmla v22.8h, v11.8h, v2.h[5]\n"
        "fmla v23.8h, v11.8h, v3.h[5]\n"

        // lane 6
        "ldr q8, [x8, #96]\n"
        "ldr q9, [x9, #96]\n"
        "fmla v16.8h, v8.8h, v0.h[6]\n"
        "fmla v17.8h, v8.8h, v1.h[6]\n"
        "fmla v18.8h, v8.8h, v2.h[6]\n"
        "fmla v19.8h, v8.8h, v3.h[6]\n"
        "fmla v20.8h, v9.8h, v0.h[6]\n"
        "fmla v21.8h, v9.8h, v1.h[6]\n"
        "fmla v22.8h, v9.8h, v2.h[6]\n"
        "fmla v23.8h, v9.8h, v3.h[6]\n"

        // lane 7
        "ldr q10, [x8, #112]\n"
        "ldr q11, [x9, #112]\n"
        "fmla v16.8h, v10.8h, v0.h[7]\n"
        "fmla v17.8h, v10.8h, v1.h[7]\n"
        "fmla v18.8h, v10.8h, v2.h[7]\n"
        "fmla v19.8h, v10.8h, v3.h[7]\n"
        "fmla v20.8h, v11.8h, v0.h[7]\n"
        "fmla v21.8h, v11.8h, v1.h[7]\n"
        "fmla v22.8h, v11.8h, v2.h[7]\n"
        "fmla v23.8h, v11.8h, v3.h[7]\n"

        // Advance to next 8-K block.
        "add x8, x8, #128\n"
        "add x9, x9, #128\n"
        "add x7, x7, #16\n"

        "subs x14, x14, #1\n"
        "b.eq 6f\n"

        // Preload next block's lane 0 A so next iteration starts ready.
        "ldr q8, [x8]\n"
        "ldr q9, [x9]\n"
        "b 3b\n"

        // ================================================================
        // Scalar tail: if K is not multiple of 8.
        // x7 is current B byte offset.
        // x8/x9 already point to matching packed-A offset.
        // ================================================================
        "6:\n"
        "cmp x7, %[K2]\n"
        "b.hs 7f\n"

        "5:\n"
        "ldr h0, [x10, x7]\n"
        "ldr h1, [x11, x7]\n"
        "ldr h2, [x12, x7]\n"
        "ldr h3, [x13, x7]\n"

        "ldr q8, [x8]\n"
        "ldr q9, [x9]\n"

        "fmla v16.8h, v8.8h, v0.h[0]\n"
        "fmla v17.8h, v8.8h, v1.h[0]\n"
        "fmla v18.8h, v8.8h, v2.h[0]\n"
        "fmla v19.8h, v8.8h, v3.h[0]\n"
        "fmla v20.8h, v9.8h, v0.h[0]\n"
        "fmla v21.8h, v9.8h, v1.h[0]\n"
        "fmla v22.8h, v9.8h, v2.h[0]\n"
        "fmla v23.8h, v9.8h, v3.h[0]\n"

        "add x7, x7, #2\n"
        "add x8, x8, #16\n"
        "add x9, x9, #16\n"
        "cmp x7, %[K2]\n"
        "b.lo 5b\n"

        // ================================================================
        // Store 4 x 16 FP32
        // ================================================================
        "7:\n"
        "lsl x5, x2, #2\n"
        "add x5, x5, %[C]\n"

#define STORE_16X4_ROW(LO, HI)                                      \
        "fcvtl v0.4s, v" #LO ".4h\n"                               \
        "fcvtl2 v1.4s, v" #LO ".8h\n"                              \
        "stp q0, q1, [x5, #0]\n"                                   \
        "fcvtl v0.4s, v" #HI ".4h\n"                               \
        "fcvtl2 v1.4s, v" #HI ".8h\n"                              \
        "stp q0, q1, [x5, #32]\n"

        STORE_16X4_ROW(16, 20)
        "add x5, x5, %[c_stride]\n"

        STORE_16X4_ROW(17, 21)
        "add x5, x5, %[c_stride]\n"

        STORE_16X4_ROW(18, 22)
        "add x5, x5, %[c_stride]\n"

        STORE_16X4_ROW(19, 23)

#undef STORE_16X4_ROW

        // Next M tile.
        "add x3, x3, %[a_stride_lsl4]\n"
        "add x2, x2, #16\n"
        "cmp x2, %[M]\n"
        "b.cc 2b\n"

        // Next N tile, 4 rows.
        "add %[C], %[C], %[c_stride_lsl2]\n"
        "add x10, x10, %[b_stride_lsl2]\n"

        "add x1, x1, #4\n"
        "cmp x1, %[N]\n"
        "b.cc 1b\n"

        : [C] "+r"(C)
        : [N] "r"(N),
          [M] "r"(M),
          [K2] "r"(K * 2),
          [B] "r"(B),
          [pa] "r"(pack_a->p),
          [a_stride_lsl3] "r"(a_stride_bytes * 8),
          [a_stride_lsl4] "r"(a_stride_bytes * 16),
          [b_stride] "r"(b_stride_bytes),
          [b_stride_lsl2] "r"(b_stride_bytes * 4),
          [c_stride] "r"(c_stride_bytes),
          [c_stride_lsl2] "r"(c_stride_bytes * 4)
        : "memory", "cc",
          "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
          "x10", "x11", "x12", "x13", "x14",
          "v0", "v1", "v2", "v3",
          "v8", "v9", "v10", "v11",
          "v16", "v17", "v18", "v19",
          "v20", "v21", "v22", "v23"
    );
}
// Kunpeng 950 software-pipelined variant:
// - Uses v16-v31 for all 16 accumulators, so only v8-v11 are callee-saved.
//   This removes the save/restore of d12-d15 from the generated prologue/epilogue.
// - Contains no software PRFM. Verify the rebuilt disassembly also contains none.
// - Removes the 10 inner-loop software prefetch instructions.
// - Delays the second packed-A half load and overlaps it with first-half FMLA.
// - Splits B loads into 4+4 groups and interleaves them with FMLA.
// - Reuses packed-A addresses for the second 8K sub-block.
// - Uses a precomputed countdown for the 32-byte main K loop, eliminating
//   the per-iteration add/cmp/exit-branch sequence.
// - Keeps arithmetic order, tile shape, packing format and tail handling.
void
matmul_outer_8x16_b_micro_packA_f16(size_t M,
                                    size_t N,
                                    size_t K,
                                    const void* a,
                                    size_t a_column_bytes,
                                    const void* b,
                                    size_t b_column_bytes,
                                    void* GGML_RESTRICT c,
                                    size_t c_column_bytes)
{
    GGML_UNUSED(a_column_bytes);
    const matmul_outer_pack_t* pack_a = (const matmul_outer_pack_t*)a;
    assert(M % 16 == 0 && N % 8 == 0);
    float32_t* GGML_RESTRICT C = (float32_t*)(c);
    const float16_t* GGML_RESTRICT B = (const float16_t*)(b);
    size_t c_stride = c_column_bytes / sizeof(float32_t);
    size_t b_stride = b_column_bytes / sizeof(float16_t);
    size_t a_stride = pack_a->p_stride / sizeof(float16_t);

    const size_t a_stride_bytes = a_stride * sizeof(float16_t);
    const size_t b_stride_bytes = b_stride * sizeof(float16_t);
    const size_t c_stride_bytes = c_stride * sizeof(float32_t);

    asm volatile(
            "mov x1, #0\n"                     // i = 0
            "mov x10, %[B]\n"                  // pb0 = &B[0]

            "1:\n"                             // for i..N, step=8
            "add x11, x10, %[b_stride]\n"      // pb1 = pb0 + b_stride
            "add x12, x11, %[b_stride]\n"
            "add x13, x12, %[b_stride]\n"
            "add x14, x13, %[b_stride]\n"
            "add x15, x14, %[b_stride]\n"
            "add x16, x15, %[b_stride]\n"
            "add x17, x16, %[b_stride]\n"

            "mov x2, #0\n"                     // j = 0
            "mov x3, %[pa]\n"                  // pa0 = pa

            "2:\n"                             // for j..M, step=16
            "add x4, x3, %[a_stride_lsl3]\n"   // pa1 = pa0 + 8 * a_stride_bytes

            "movi v16.16b, #0\n"
            "movi v17.16b, #0\n"
            "movi v18.16b, #0\n"
            "movi v19.16b, #0\n"
            "movi v20.16b, #0\n"
            "movi v21.16b, #0\n"
            "movi v22.16b, #0\n"
            "movi v23.16b, #0\n"
            "movi v24.16b, #0\n"
            "movi v25.16b, #0\n"
            "movi v26.16b, #0\n"
            "movi v27.16b, #0\n"
            "movi v28.16b, #0\n"
            "movi v29.16b, #0\n"
            "movi v30.16b, #0\n"
            "movi v31.16b, #0\n"

            "mov x7, #0\n"                     // k2 = byte offset into B
            "lsr x18, %[K2], #5\n"            // number of complete 32-byte K chunks
            "cbz x18, 7f\n"

            // ==== main unroll2 loop: exactly floor(K2 / 32) iterations ====
            "3:\n"

            // Inner-loop PRFM removed permanently to reduce frontend/LSU pressure.

#define KERNEL_8x8_M_D(idx, rl1, rl2, rc1, rc2)                    \
            "ldr q" #rl1 ", [x8, #((" #idx " + 1) * 16)]\n"      \
            "ldr q" #rl2 ", [x9, #((" #idx " + 1) * 16)]\n"      \
            /* Interleave low/high M accumulators to spread rename/issue pressure. */ \
            "fmla v16.8h, v" #rc1 ".8h, v0.h[" #idx "]\n"       \
            "fmla v24.8h, v" #rc2 ".8h, v0.h[" #idx "]\n"       \
            "fmla v17.8h, v" #rc1 ".8h, v1.h[" #idx "]\n"       \
            "fmla v25.8h, v" #rc2 ".8h, v1.h[" #idx "]\n"       \
            "fmla v18.8h, v" #rc1 ".8h, v2.h[" #idx "]\n"       \
            "fmla v26.8h, v" #rc2 ".8h, v2.h[" #idx "]\n"       \
            "fmla v19.8h, v" #rc1 ".8h, v3.h[" #idx "]\n"       \
            "fmla v27.8h, v" #rc2 ".8h, v3.h[" #idx "]\n"       \
            "fmla v20.8h, v" #rc1 ".8h, v4.h[" #idx "]\n"       \
            "fmla v28.8h, v" #rc2 ".8h, v4.h[" #idx "]\n"       \
            "fmla v21.8h, v" #rc1 ".8h, v5.h[" #idx "]\n"       \
            "fmla v29.8h, v" #rc2 ".8h, v5.h[" #idx "]\n"       \
            "fmla v22.8h, v" #rc1 ".8h, v6.h[" #idx "]\n"       \
            "fmla v30.8h, v" #rc2 ".8h, v6.h[" #idx "]\n"       \
            "fmla v23.8h, v" #rc1 ".8h, v7.h[" #idx "]\n"       \
            "fmla v31.8h, v" #rc2 ".8h, v7.h[" #idx "]\n"

#define KERNEL_BODY_8K_D()                                          \
            /* Load the first M-half only; delay the second M-half load. */ \
            "ldp q8, q10, [x8, #(0 * 16)]\n"                        \
            "ldr q0, [x10, x6]\n"                                   \
            "ldr q1, [x11, x6]\n"                                   \
            "ldr q2, [x12, x6]\n"                                   \
            "ldr q3, [x13, x6]\n"                                   \
            "fmla v16.8h, v8.8h, v0.h[0]\n"                        \
            "fmla v17.8h, v8.8h, v1.h[0]\n"                        \
            "fmla v18.8h, v8.8h, v2.h[0]\n"                        \
            "fmla v19.8h, v8.8h, v3.h[0]\n"                        \
            /* q9/q11 now overlap with first-half FMLA retirement. */ \
            "ldp q9, q11, [x9, #(0 * 16)]\n"                        \
            "fmla v24.8h, v9.8h, v0.h[0]\n"                        \
            "fmla v25.8h, v9.8h, v1.h[0]\n"                        \
            "fmla v26.8h, v9.8h, v2.h[0]\n"                        \
            "fmla v27.8h, v9.8h, v3.h[0]\n"                        \
            "ldr q4, [x14, x6]\n"                                   \
            "ldr q5, [x15, x6]\n"                                   \
            "ldr q6, [x16, x6]\n"                                   \
            "ldr q7, [x17, x6]\n"                                   \
            "fmla v20.8h, v8.8h, v4.h[0]\n"                        \
            "fmla v21.8h, v8.8h, v5.h[0]\n"                        \
            "fmla v22.8h, v8.8h, v6.h[0]\n"                        \
            "fmla v23.8h, v8.8h, v7.h[0]\n"                        \
            "fmla v28.8h, v9.8h, v4.h[0]\n"                        \
            "fmla v29.8h, v9.8h, v5.h[0]\n"                        \
            "fmla v30.8h, v9.8h, v6.h[0]\n"                        \
            "fmla v31.8h, v9.8h, v7.h[0]\n"                        \
            KERNEL_8x8_M_D(1,  8,  9, 10, 11)                       \
            KERNEL_8x8_M_D(2, 10, 11,  8,  9)                       \
            KERNEL_8x8_M_D(3,  8,  9, 10, 11)                       \
            KERNEL_8x8_M_D(4, 10, 11,  8,  9)                       \
            KERNEL_8x8_M_D(5,  8,  9, 10, 11)                       \
            KERNEL_8x8_M_D(6, 10, 11,  8,  9)                       \
            "fmla v16.8h, v10.8h, v0.h[7]\n"                       \
            "fmla v17.8h, v10.8h, v1.h[7]\n"                       \
            "fmla v18.8h, v10.8h, v2.h[7]\n"                       \
            "fmla v19.8h, v10.8h, v3.h[7]\n"                       \
            "fmla v20.8h, v10.8h, v4.h[7]\n"                       \
            "fmla v21.8h, v10.8h, v5.h[7]\n"                       \
            "fmla v22.8h, v10.8h, v6.h[7]\n"                       \
            "fmla v23.8h, v10.8h, v7.h[7]\n"                       \
            "fmla v24.8h, v11.8h, v0.h[7]\n"                       \
            "fmla v25.8h, v11.8h, v1.h[7]\n"                       \
            "fmla v26.8h, v11.8h, v2.h[7]\n"                       \
            "fmla v27.8h, v11.8h, v3.h[7]\n"                       \
            "fmla v28.8h, v11.8h, v4.h[7]\n"                       \
            "fmla v29.8h, v11.8h, v5.h[7]\n"                       \
            "fmla v30.8h, v11.8h, v6.h[7]\n"                       \
            "fmla v31.8h, v11.8h, v7.h[7]\n"

            // First 8K sub-block: calculate packed-A addresses once.
            "lsl x6, x7, #3\n"
            "add x8, x3, x6\n"
            "add x9, x4, x6\n"
            "mov x6, x7\n"
            KERNEL_BODY_8K_D()

            // Second 8K sub-block: fixed increments replace add/lsl/add.
            "add x8, x8, #128\n"
            "add x9, x9, #128\n"
            "add x6, x7, #16\n"
            KERNEL_BODY_8K_D()

#undef KERNEL_BODY_8K_D
#undef KERNEL_8x8_M_D

            "add x7, x7, #32\n"
            "subs x18, x18, #1\n"
            "b.ne 3b\n"

            // ==== remainder full block loop: for (; k2 + 16 <= K2; k2 += 16) ====
            "7:\n"
            "add x6, x7, #16\n"
            "cmp x6, %[K2]\n"
            "b.hi 4f\n"

            "lsl x6, x7, #3\n"
            "add x8, x3, x6\n"
            "add x9, x4, x6\n"
            "ldp q8, q10, [x8, #(0 * 16)]\n"
            "ldr q0, [x10, x7]\n"
            "ldr q1, [x11, x7]\n"
            "ldr q2, [x12, x7]\n"
            "ldr q3, [x13, x7]\n"
            "fmla v16.8h, v8.8h, v0.h[0]\n"
            "fmla v17.8h, v8.8h, v1.h[0]\n"
            "fmla v18.8h, v8.8h, v2.h[0]\n"
            "fmla v19.8h, v8.8h, v3.h[0]\n"
            "ldp q9, q11, [x9, #(0 * 16)]\n"
            "fmla v24.8h, v9.8h, v0.h[0]\n"
            "fmla v25.8h, v9.8h, v1.h[0]\n"
            "fmla v26.8h, v9.8h, v2.h[0]\n"
            "fmla v27.8h, v9.8h, v3.h[0]\n"
            "ldr q4, [x14, x7]\n"
            "ldr q5, [x15, x7]\n"
            "ldr q6, [x16, x7]\n"
            "ldr q7, [x17, x7]\n"
            "fmla v20.8h, v8.8h, v4.h[0]\n"
            "fmla v21.8h, v8.8h, v5.h[0]\n"
            "fmla v22.8h, v8.8h, v6.h[0]\n"
            "fmla v23.8h, v8.8h, v7.h[0]\n"
            "fmla v28.8h, v9.8h, v4.h[0]\n"
            "fmla v29.8h, v9.8h, v5.h[0]\n"
            "fmla v30.8h, v9.8h, v6.h[0]\n"
            "fmla v31.8h, v9.8h, v7.h[0]\n"

#define KERNEL_8x8_M_D_REM(idx, rl1, rl2, rc1, rc2)                    \
            "ldr q" #rl1 ", [x8, #((" #idx " + 1) * 16)]\n"      \
            "ldr q" #rl2 ", [x9, #((" #idx " + 1) * 16)]\n"      \
            /* Interleave low/high M accumulators to spread rename/issue pressure. */ \
            "fmla v16.8h, v" #rc1 ".8h, v0.h[" #idx "]\n"       \
            "fmla v24.8h, v" #rc2 ".8h, v0.h[" #idx "]\n"       \
            "fmla v17.8h, v" #rc1 ".8h, v1.h[" #idx "]\n"       \
            "fmla v25.8h, v" #rc2 ".8h, v1.h[" #idx "]\n"       \
            "fmla v18.8h, v" #rc1 ".8h, v2.h[" #idx "]\n"       \
            "fmla v26.8h, v" #rc2 ".8h, v2.h[" #idx "]\n"       \
            "fmla v19.8h, v" #rc1 ".8h, v3.h[" #idx "]\n"       \
            "fmla v27.8h, v" #rc2 ".8h, v3.h[" #idx "]\n"       \
            "fmla v20.8h, v" #rc1 ".8h, v4.h[" #idx "]\n"       \
            "fmla v28.8h, v" #rc2 ".8h, v4.h[" #idx "]\n"       \
            "fmla v21.8h, v" #rc1 ".8h, v5.h[" #idx "]\n"       \
            "fmla v29.8h, v" #rc2 ".8h, v5.h[" #idx "]\n"       \
            "fmla v22.8h, v" #rc1 ".8h, v6.h[" #idx "]\n"       \
            "fmla v30.8h, v" #rc2 ".8h, v6.h[" #idx "]\n"       \
            "fmla v23.8h, v" #rc1 ".8h, v7.h[" #idx "]\n"       \
            "fmla v31.8h, v" #rc2 ".8h, v7.h[" #idx "]\n"

            KERNEL_8x8_M_D_REM(1,  8,  9, 10, 11)
            KERNEL_8x8_M_D_REM(2, 10, 11,  8,  9)
            KERNEL_8x8_M_D_REM(3,  8,  9, 10, 11)
            KERNEL_8x8_M_D_REM(4, 10, 11,  8,  9)
            KERNEL_8x8_M_D_REM(5,  8,  9, 10, 11)
            KERNEL_8x8_M_D_REM(6, 10, 11,  8,  9)
#undef KERNEL_8x8_M_D_REM

            "fmla v16.8h, v10.8h, v0.h[7]\n"
            "fmla v24.8h, v11.8h, v0.h[7]\n"
            "fmla v17.8h, v10.8h, v1.h[7]\n"
            "fmla v25.8h, v11.8h, v1.h[7]\n"
            "fmla v18.8h, v10.8h, v2.h[7]\n"
            "fmla v26.8h, v11.8h, v2.h[7]\n"
            "fmla v19.8h, v10.8h, v3.h[7]\n"
            "fmla v27.8h, v11.8h, v3.h[7]\n"
            "fmla v20.8h, v10.8h, v4.h[7]\n"
            "fmla v28.8h, v11.8h, v4.h[7]\n"
            "fmla v21.8h, v10.8h, v5.h[7]\n"
            "fmla v29.8h, v11.8h, v5.h[7]\n"
            "fmla v22.8h, v10.8h, v6.h[7]\n"
            "fmla v30.8h, v11.8h, v6.h[7]\n"
            "fmla v23.8h, v10.8h, v7.h[7]\n"
            "fmla v31.8h, v11.8h, v7.h[7]\n"
            "add x7, x7, #16\n"
            "b 7b\n"

            "4:\n"                             // end full-block K loops

            // ==== tail: for (; k2 < K2; k2 += 2), one fp16 at a time ====
            "5:\n"
            "cmp x7, %[K2]\n"
            "b.hs 6f\n"

            "lsl x8, x7, #3\n"
            "ldr h0, [x10, x7]\n"
            "ldr h1, [x11, x7]\n"
            "ldr h2, [x12, x7]\n"
            "ldr h3, [x13, x7]\n"
            "ldr h4, [x14, x7]\n"
            "ldr h5, [x15, x7]\n"
            "ldr h6, [x16, x7]\n"
            "ldr h7, [x17, x7]\n"
            "ldr q8, [x3, x8]\n"
            "ldr q9, [x4, x8]\n"
            "fmla v16.8h, v8.8h, v0.h[0]\n"
            "fmla v17.8h, v8.8h, v1.h[0]\n"
            "fmla v18.8h, v8.8h, v2.h[0]\n"
            "fmla v19.8h, v8.8h, v3.h[0]\n"
            "fmla v20.8h, v8.8h, v4.h[0]\n"
            "fmla v21.8h, v8.8h, v5.h[0]\n"
            "fmla v22.8h, v8.8h, v6.h[0]\n"
            "fmla v23.8h, v8.8h, v7.h[0]\n"
            "fmla v24.8h, v9.8h, v0.h[0]\n"
            "fmla v25.8h, v9.8h, v1.h[0]\n"
            "fmla v26.8h, v9.8h, v2.h[0]\n"
            "fmla v27.8h, v9.8h, v3.h[0]\n"
            "fmla v28.8h, v9.8h, v4.h[0]\n"
            "fmla v29.8h, v9.8h, v5.h[0]\n"
            "fmla v30.8h, v9.8h, v6.h[0]\n"
            "fmla v31.8h, v9.8h, v7.h[0]\n"

            "add x7, x7, #2\n"
            "b 5b\n"

            "6:\n"                             // end K loop

            "add x3, x3, %[a_stride_lsl4]\n"   // pa0 += 16 * a_stride_bytes

            "lsl x5, x2, #2\n"
            "add x5, x5, %[C]\n"               // ptrC = C + j

            "fcvtl v0.4s, v16.4h\n  fcvtl2 v1.4s, v16.8h\n  stp q0, q1, [x5, #0]\n"
            "fcvtl v0.4s, v24.4h\n  fcvtl2 v1.4s, v24.8h\n  stp q0, q1, [x5, #32]\n"
            "add x5, x5, %[c_stride]\n"
            "fcvtl v0.4s, v17.4h\n  fcvtl2 v1.4s, v17.8h\n  stp q0, q1, [x5, #0]\n"
            "fcvtl v0.4s, v25.4h\n  fcvtl2 v1.4s, v25.8h\n  stp q0, q1, [x5, #32]\n"
            "add x5, x5, %[c_stride]\n"
            "fcvtl v0.4s, v18.4h\n  fcvtl2 v1.4s, v18.8h\n  stp q0, q1, [x5, #0]\n"
            "fcvtl v0.4s, v26.4h\n  fcvtl2 v1.4s, v26.8h\n  stp q0, q1, [x5, #32]\n"
            "add x5, x5, %[c_stride]\n"
            "fcvtl v0.4s, v19.4h\n  fcvtl2 v1.4s, v19.8h\n  stp q0, q1, [x5, #0]\n"
            "fcvtl v0.4s, v27.4h\n  fcvtl2 v1.4s, v27.8h\n  stp q0, q1, [x5, #32]\n"
            "add x5, x5, %[c_stride]\n"
            "fcvtl v0.4s, v20.4h\n  fcvtl2 v1.4s, v20.8h\n  stp q0, q1, [x5, #0]\n"
            "fcvtl v0.4s, v28.4h\n  fcvtl2 v1.4s, v28.8h\n  stp q0, q1, [x5, #32]\n"
            "add x5, x5, %[c_stride]\n"
            "fcvtl v0.4s, v21.4h\n  fcvtl2 v1.4s, v21.8h\n  stp q0, q1, [x5, #0]\n"
            "fcvtl v0.4s, v29.4h\n  fcvtl2 v1.4s, v29.8h\n  stp q0, q1, [x5, #32]\n"
            "add x5, x5, %[c_stride]\n"
            "fcvtl v0.4s, v22.4h\n  fcvtl2 v1.4s, v22.8h\n  stp q0, q1, [x5, #0]\n"
            "fcvtl v0.4s, v30.4h\n  fcvtl2 v1.4s, v30.8h\n  stp q0, q1, [x5, #32]\n"
            "add x5, x5, %[c_stride]\n"
            "fcvtl v0.4s, v23.4h\n  fcvtl2 v1.4s, v23.8h\n  stp q0, q1, [x5, #0]\n"
            "fcvtl v0.4s, v31.4h\n  fcvtl2 v1.4s, v31.8h\n  stp q0, q1, [x5, #32]\n"

            "add x2, x2, #16\n"
            "cmp x2, %[M]\n"
            "b.cc 2b\n"

            "add %[C], %[C], %[c_stride_lsl3]\n" // C += 8 * c_stride_bytes
            "add x10, x10, %[b_stride_lsl3]\n"  // pb0 += 8 * b_stride_bytes

            "add x1, x1, #8\n"
            "cmp x1, %[N]\n"
            "b.cc 1b\n"
        : [C] "+r"(C)
        : [N] "r"(N),
          [M] "r"(M),
          [K2] "r"(K * 2),
          [B] "r"(B),
          [pa] "r"(pack_a->p),
          [a_stride_lsl3] "r"(a_stride_bytes * 8),
          [a_stride_lsl4] "r"(a_stride_bytes * 16),
          [b_stride] "r"(b_stride_bytes),
          [b_stride_lsl3] "r"(b_stride_bytes * 8),
          [c_stride] "r"(c_stride_bytes),
          [c_stride_lsl3] "r"(c_stride_bytes * 8)
        : "memory", "cc",
          "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
          "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
    );
}
void matmul_outer_packA_destroy(matmul_outer_pack_t *spack) {
    if (!spack) {
        return;
    }
    if (spack->p) {
        free(spack->p);
    }
    free(spack);
}

void
matmul_outer_packA(size_t nr,
                  int dim,
                  const void* GGML_RESTRICT X_,
                  size_t el_bits,
                  size_t column_bytes,
                  matmul_outer_pack_t** out)
{
    const uint8_t* GGML_RESTRICT X = (const uint8_t*)X_;
    const size_t K = dim;
    size_t el_bytes = el_bits / 8;
    size_t K_bytes = K * el_bytes;
    size_t stride_8b = column_bytes / sizeof(uint8_t);
    assert(X != NULL);
    matmul_outer_pack_t* pack = NULL;
    size_t sz_p = MAX(sizeof(*pack->p) * nr * K_bytes, 64);
    if (*out != NULL && (*out)->sz_p == sz_p && (*out)->nr == nr && (*out)->dim == dim) {
        pack = *out;
    } else {
        if (*out) {
            matmul_outer_packA_destroy(*out);
            *out = NULL;
        }
        pack = (matmul_outer_pack_t*)malloc(sizeof(matmul_outer_pack_t));
        pack->p = (uint8_t*)aligned_alloc(64, sz_p);
        pack->p_stride = K_bytes;
        pack->sz_p = sz_p;
        pack->nr = nr;
        pack->dim = dim;
    }
    // NOTE: Assume that we are using SVE256
    for (size_t i = 0; i < nr;) {
        if (el_bits == 32) {
#if 0
            svuint32_t ldst_indices = svindex_u32(0, stride_8b / sizeof(uint32_t));
            svbool_t pg = svwhilelt_b32(i, MIN(nr, i + 4));
            svbool_t pg_vl4 = svptrue_pat_b32(SV_VL4);
            for (size_t ki = 0; ki < K; ki++) {
                svuint32_t svb = svld1_gather_u32index_u32(pg, (const uint32_t*)&X[(i + 0) * stride_8b + ki * sizeof(uint32_t)], ldst_indices);
                svst1_u32(pg_vl4, (uint32_t*)&pack->p[i * pack->p_stride + ki * sizeof(uint32_t) * 4], svb);
            }
#else
            const size_t vl_32b = svcntw();
            for (size_t ki = 0; ki < K; ki += vl_32b) {
                svuint32_t row0, row1, row2, row3;
                svbool_t pg = svwhilelt_b32(ki, K);
                const static uint32_t zero_x[8] = { 0 };
                const uint32_t* ptr_rows[4];
                for (size_t t = 0; t < 4; t++) {
                    ptr_rows[t] =
                        i + t < nr ? (const uint32_t*)&X[(i + t) * stride_8b + ki * sizeof(uint32_t)] : zero_x;
                }
                row0 = svld1_u32(pg, ptr_rows[0]);
                row1 = svld1_u32(pg, ptr_rows[1]);
                row2 = svld1_u32(pg, ptr_rows[2]);
                row3 = svld1_u32(pg, ptr_rows[3]);
                svst4_u32(pg,
                            (uint32_t*)&pack->p[i * pack->p_stride + ki * sizeof(uint32_t) * 4],
                            svcreate4_u32(row0, row1, row2, row3));
            }
#endif
            i += 4;
        } else if(el_bits == 16) {
            if (K < svcnth()) {
                svuint32_t ldst_indices = svindex_u32(0, stride_8b / sizeof(uint16_t));
                svbool_t pg = svwhilelt_b32(i, MIN(nr, i + 8));
                svbool_t pg_vl8 = svptrue_pat_b16(SV_VL8);
                for (size_t ki = 0; ki < K; ki++) {
                    svuint32_t svbl = svld1uh_gather_u32index_u32(pg, (const uint16_t*)&X[(i + 0) * stride_8b + ki * sizeof(uint16_t)], ldst_indices);
                    svuint16_t svb = svuzp1_u16(svreinterpret_u16_u32(svbl), svdup_u16(0));
                    svst1_u16(pg_vl8, (uint16_t*)&pack->p[i * pack->p_stride + ki * sizeof(uint16_t) * 8], svb);
                }
            } else {
                const size_t vl_16b = svcnth();
                for (size_t ki = 0; ki < K; ki += vl_16b) {
                    svuint16_t row0, row1, row2, row3;
                    svuint16_t row4, row5, row6, row7;
                    svbool_t pg = svwhilelt_b16(ki, K);
                    const static uint16_t zero_x[16] = { 0 };
                    const uint16_t* ptr_rows[8];
                    for (size_t t = 0; t < 8; t++) {
                        ptr_rows[t] = i + t < nr ? (const uint16_t*)&X[(i + t) * stride_8b + ki * sizeof(uint16_t)] : zero_x;
                    }
                    row0 = svld1_u16(pg, ptr_rows[0]);
                    row1 = svld1_u16(pg, ptr_rows[1]);
                    row2 = svld1_u16(pg, ptr_rows[2]);
                    row3 = svld1_u16(pg, ptr_rows[3]);
                    row4 = svld1_u16(pg, ptr_rows[4]);
                    row5 = svld1_u16(pg, ptr_rows[5]);
                    row6 = svld1_u16(pg, ptr_rows[6]);
                    row7 = svld1_u16(pg, ptr_rows[7]);
                    svuint16_t zip1_04 = svzip1_u16(row0, row4);
                    svuint16_t zip1_15 = svzip1_u16(row1, row5);
                    svuint16_t zip1_26 = svzip1_u16(row2, row6);
                    svuint16_t zip1_37 = svzip1_u16(row3, row7);
                    svst4_u16(pg,
                            (uint16_t*)&pack
                                ->p[i * pack->p_stride + ki * sizeof(uint16_t) * 8 + 0 * 8 * 8 * sizeof(uint16_t)],
                            svcreate4_u16(zip1_04, zip1_15, zip1_26, zip1_37));
                    svuint16_t zip2_04 = svzip2_u16(row0, row4);
                    svuint16_t zip2_15 = svzip2_u16(row1, row5);
                    svuint16_t zip2_26 = svzip2_u16(row2, row6);
                    svuint16_t zip2_37 = svzip2_u16(row3, row7);
                    svst4_u16(pg,
                            (uint16_t*)&pack
                                ->p[i * pack->p_stride + ki * sizeof(uint16_t) * 8 + 1 * 8 * 8 * sizeof(uint16_t)],
                            svcreate4_u16(zip2_04, zip2_15, zip2_26, zip2_37));
                }
            }
            i += 8;
        }
    }
    *out = pack;
}

void
matmul_outer_packA_b(size_t na,
                     size_t nb,
                     int dim,
                     size_t el_bits,
                     const void* GGML_RESTRICT a,
                     const void* GGML_RESTRICT b,
                     void* GGML_RESTRICT c,
                     size_t c_column_bytes,
                     matmul_outer_packA_func func)
{
    assert(func);
    const size_t M = na;
    const size_t N = nb;
    const size_t K = dim;
    const size_t el_bytes = el_bits / 8;
    const uint8_t* GGML_RESTRICT A = (const uint8_t*)(a);
    const uint8_t* GGML_RESTRICT B = (const uint8_t*)(b);
    float32_t* GGML_RESTRICT C = (float32_t*)(c);
    size_t c_stride = c_column_bytes / sizeof(float32_t);
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            C[i * c_stride + j] = 0.0f;
        }
    }
    if (M == 0 || N == 0) return;
    size_t M_BLK = g_chunk_size_outer_packA / 4;
    size_t N_BLK = g_chunk_size_outer_packA / 1;
    size_t K_BLK = 4096;
    size_t a_column_bytes = K * el_bytes;
    size_t b_column_bytes = K * el_bytes;
    matmul_outer_pack_t* pack_a = NULL;
    for (size_t ki = 0; ki < K; ki += K_BLK) {
        size_t k = MIN(K_BLK, K - ki);
        size_t kb = ki * el_bits / 8;
        for (size_t m = 0; m < M; m += M_BLK) {
            size_t na = MIN(M_BLK, M - m);
            matmul_outer_packA(na, k, &A[m * a_column_bytes + kb], el_bits, a_column_bytes, &pack_a);
            for (size_t n = 0; n < N; n += N_BLK) {
                size_t nb = MIN(N_BLK, N - n);
                func(na, nb, k, pack_a, pack_a->p_stride, &B[n * b_column_bytes + kb], b_column_bytes, &C[n * c_stride + m], c_column_bytes);
            }
        }
    }
    matmul_outer_packA_destroy(pack_a);
}

void ggml_vec_silu_f32(const int n, float * y, const float * x) {
    int i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i, ggml_v_silu(_mm512_loadu_ps(x + i)));
    }
#elif defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i, ggml_v_silu(_mm256_loadu_ps(x + i)));
    }
#elif defined(__SSE2__)
    for (; i + 3 < n; i += 4) {
        _mm_storeu_ps(y + i, ggml_v_silu(_mm_loadu_ps(x + i)));
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (; i + 3 < n; i += 4) {
        vst1q_f32(y + i, ggml_v_silu(vld1q_f32(x + i)));
    }
#endif
    for (; i < n; ++i) {
        y[i] = ggml_silu_f32(x[i]);
    }
}

ggml_float ggml_vec_soft_max_f32(const int n, float * y, const float * x, float max) {
    int i = 0;
    ggml_float sum = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i + 15 < n; i += 16) {
        __m512 val = ggml_v_expf(_mm512_sub_ps(_mm512_loadu_ps(x + i),
                                               _mm512_set1_ps(max)));
        _mm512_storeu_ps(y + i, val);
        sum += (ggml_float)_mm512_reduce_add_ps(val);
    }
#elif defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        __m256 val = ggml_v_expf(_mm256_sub_ps(_mm256_loadu_ps(x + i),
                                               _mm256_set1_ps(max)));
        _mm256_storeu_ps(y + i, val);
        __m128 val2 = _mm_add_ps(_mm256_extractf128_ps(val, 1),
                                 _mm256_castps256_ps128(val));
        val2 = _mm_add_ps(val2, _mm_movehl_ps(val2, val2));
        val2 = _mm_add_ss(val2, _mm_movehdup_ps(val2));
        sum += (ggml_float)_mm_cvtss_f32(val2);
    }
#elif defined(__SSE2__)
    for (; i + 3 < n; i += 4) {
        __m128 val = ggml_v_expf(_mm_sub_ps(_mm_loadu_ps(x + i),
                                            _mm_set1_ps(max)));
        _mm_storeu_ps(y + i, val);
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
        val = _mm_add_ps(val, _mm_movehl_ps(val, val));
        val = _mm_add_ss(val, _mm_movehdup_ps(val));
#else
        __m128 tmp = _mm_shuffle_ps(val, val, _MM_SHUFFLE(2, 3, 0, 1));
        val = _mm_add_ps(val, tmp);
        tmp = _mm_movehl_ps(tmp, val);
        val = _mm_add_ss(val, tmp);
#endif
        sum += (ggml_float)_mm_cvtss_f32(val);
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (; i + 3 < n; i += 4) {
        float32x4_t val = ggml_v_expf(vsubq_f32(vld1q_f32(x + i),
                                                vdupq_n_f32(max)));
        vst1q_f32(y + i, val);
        sum += (ggml_float)vaddvq_f32(val);
    }
#endif
    for (; i < n; ++i) {
        float val = expf(x[i] - max);
        sum += (ggml_float)val;
        y[i] = val;
    }
    return sum;
}

ggml_float ggml_vec_log_soft_max_f32(const int n, float * y, const float * x, float max) {
    // log(soft_max) = log(soft_max_i / soft_max_sum) = log(soft_max_i) - log(soft_max_sum) = (logit_i - max) - log(soft_max_i)

    int i = 0;
    ggml_float sum = 0;
    for (; i < n; ++i) {
        float val = x[i] - max;
        y[i] = val;
        sum += (ggml_float)expf(val);
    }
    return sum = (ggml_float)logf(sum);
}
