#include <cmath>
#include <iostream>

#include "utils.h"
#include "modules.h"


#define GTEN_CHECK_DTYPE_EQUAL(inp_dtype, expected_dtype)     \
    GTEN_ASSERT(                                              \
        inp_dtype == expected_dtype,                          \
        "Expected tensor to have dtype=%s but got dtype=%s.", \
        dtype_str(expected_dtype),                            \
        dtype_str(inp_dtype))

#define GTEN_CHECK_NDIMS_EQUAL(inp_ndims, expected_ndims)    \
    GTEN_ASSERT(                                             \
        inp_ndims == expected_ndims,                         \
        "Expected a %d-dim tensor but got a %d-dim tensor.", \
        expected_ndims,                                      \
        inp_ndims)

#define GTEN_CHECK_DIMSIZE_EQUAL(dim, inp_dimsize, expected_dimsize)  \
    GTEN_ASSERT(                                                      \
        inp_dimsize == expected_dimsize,                              \
        "Expected tensor to have dim-%d=%d but got dim-%d=%d.",       \
        dim, expected_dimsize, dim, inp_dimsize)

// #define GTEN_CHECK_INP_CTX_SIZE(inp_ctx_size, max_ctx_size)                  \
//     GTEN_ASSERT(                                                             \
//         inp_ctx_size <= max_ctx_size,                                        \
//         "The given input's context size=%d exceeds max context size of %d.", \
//         inp_ctx_size,                                                        \
//         max_ctx_size)

#define GTEN_CHECK_INP_CTX_SIZE(inp_ctx_size, max_ctx_size)



namespace gten
{

Embedding::Embedding(const Tensor& weight, const Tensor& emb_acv, const Tensor& proj_acv)
    : weight_{weight}, emb_acv_{emb_acv}, proj_acv_{proj_acv}
{
    GTEN_CHECK_NDIMS_EQUAL(weight_.ndims(), 2);
    GTEN_CHECK_NDIMS_EQUAL(emb_acv_.ndims(), 2);
    GTEN_CHECK_NDIMS_EQUAL(proj_acv_.ndims(), 1);
    GTEN_ASSERT(
        proj_acv_.dtype() == kFloat32,
        "proj_acv tensor is expected to have dtype Float32 but got %s instead.",
        dtype_str(proj_acv_.dtype()));
}

Tensor Embedding::forward(const Tensor& inp)
{
    Timer timer{&time_embed_ms_};

    GTEN_CHECK_DTYPE_EQUAL(inp.dtype(), kInt32);
    GTEN_CHECK_NDIMS_EQUAL(inp.ndims(), 1);
    int max_ctx = emb_acv_.size(0);
    GTEN_CHECK_INP_CTX_SIZE(inp.numel(), max_ctx);
    
    return forward_impl(inp);
}

Tensor Embedding::forward_impl(const Tensor& inp)
{
    const int n_embed = weight_.size(1);
    emb_acv_.resize({inp.numel(), n_embed});
    Float16* out_data = emb_acv_.data_ptr<Float16>();
    const Int32* inp_data = inp.data_ptr<Int32>();
    Float16* weight_data = weight_.data_ptr<Float16>();

    if (emb_acv_cached_) {
        const int token_i = inp.numel() - 1;
        const int emb_i = inp_data[token_i] * n_embed;
        void *src = reinterpret_cast<void*>(weight_data + emb_i);
        void *dest = reinterpret_cast<void*>(out_data + token_i * n_embed);
        std::memcpy(dest, src, n_embed * weight_.itemsize());
    }
    else {
        emb_acv_cached_ = true;

        const int ntokens = inp.numel();
        for (int token_i = 0; token_i < ntokens; token_i++) {
            int emb_i = inp_data[token_i] * n_embed;
            void *src = reinterpret_cast<void*>(weight_data + emb_i);
            void *dest = reinterpret_cast<void*>(out_data + token_i * n_embed);
            std::memcpy(dest, src, n_embed * weight_.itemsize());
        }
    }

    return emb_acv_;    
}


Tensor Embedding::forward_proj(const Tensor &inp)
{
    Timer timer(&time_project_ms_);

    GTEN_CHECK_NDIMS_EQUAL(inp.ndims(), 2);
    GTEN_CHECK_DIMSIZE_EQUAL(1, inp.size(1), weight_.size(1));

    return forward_proj_impl(inp);
}

Tensor Embedding::forward_proj_impl(const Tensor& inp) {
    // Output probs must be float32.
    Float32* out_data = proj_acv_.data_ptr<Float32>();
    const Float16* inp_data = inp.data_ptr<Float16>();
    const Float16* weight_data = weight_.data_ptr<Float16>();

    const int n_vocab = weight_.size(0);
    const int ctx_i = inp.size(0) - 1;
    const int n_embed = inp.size(1);

    for (int emb_i = 0; emb_i < n_vocab; emb_i++)
    {
        Vec_f32x8 dot_accum = { vec_f32x8_setzero() };
        for (int i = 0; i < n_embed; i += 8) {
            Vec_f32x8 x = vec_f32x8_load(inp_data + (ctx_i * n_embed + i));
            Vec_f32x8 w = vec_f32x8_load(weight_data + (emb_i * n_embed + i));
            dot_accum = vec_f32x8_fma(x, w, dot_accum);
        }
        out_data[emb_i] = vec_f32x8_sum(dot_accum);
    }

    return proj_acv_;  
}

PosEmbedding::PosEmbedding(const Tensor& weight, int max_ctx)
    : weight_{weight}, max_ctx_(max_ctx)
{
    GTEN_CHECK_NDIMS_EQUAL(weight_.ndims(), 2);
}

Tensor PosEmbedding::forward(int n_ctx)
{
    GTEN_CHECK_INP_CTX_SIZE(n_ctx, max_ctx_);

    Timer timer{&time_ms_};
    
    return forward_impl(n_ctx);
}

Tensor PosEmbedding::forward_impl(int n_ctx)
{
    const Float16* weight_data = weight_.data_ptr<Float16>();

    void* src_ptr = (void*)weight_data;
    const int n_embed = weight_.size(1);
    Tensor acv{src_ptr, {n_ctx, n_embed}, weight_.dtype()};

    return acv;
}

LayerNorm::LayerNorm(const Tensor& weight, const Tensor& bias, const Tensor& acv)
    : weight_{weight}, bias_{bias}, acv_{acv}
{
    GTEN_CHECK_NDIMS_EQUAL(weight_.ndims(), 1);
    GTEN_CHECK_NDIMS_EQUAL(bias_.ndims(), 1);
}


Tensor LayerNorm::forward(const Tensor &inp)
{
    Timer timer(&time_ms_);

    GTEN_CHECK_NDIMS_EQUAL(inp.ndims(), 2);
    const int n_embed = weight_.size(0);
    GTEN_CHECK_DIMSIZE_EQUAL(1, inp.size(1), n_embed);

    return forward_impl(inp);
}

Tensor LayerNorm::forward_impl(const Tensor &inp)
{
    acv_.resize({inp.size(0), inp.size(1)});
    Float16* acv_data = acv_.data_ptr<Float16>();
    const Float16* inp_data = inp.data_ptr<Float16>();
    const Float16* weight_data = weight_.data_ptr<Float16>();
    const Float16* bias_data = bias_.data_ptr<Float16>();

    const int n_ctx = inp.size(0);
    const int n_embed = weight_.size(0);

    if (acv_cached_)
    {
        const int ctx_offset = (n_ctx - 1) * n_embed;
        // Mean calculation.
        Float32 mean_accum = 0.0f;
        for (int i = 0; i < n_embed; i++)
            mean_accum += fpcvt_to_fp32<Float16>(inp_data[i + ctx_offset]);
        Float32 mean = mean_accum / (Float32)n_embed;

        // Standard deviation calculation.
        Float32 variance_accum = 0.0f;
        for (int i = 0; i < n_embed; i++) {
            Float32 x = fpcvt_to_fp32<Float16>(inp_data[i + ctx_offset]);
            variance_accum += (x - mean) * (x - mean);
        }
        Float32 std_dev = std::sqrt(variance_accum / (Float32)n_embed);

        // Normalization.
        for (int i = 0; i < n_embed; i++) {
            Float32 unnormalized = fpcvt_to_fp32<Float16>(inp_data[i + ctx_offset]);
            Float32 w = fpcvt_to_fp32<Float16>(weight_data[i]);
            Float32 b = fpcvt_to_fp32<Float16>(bias_data[i]);
            Float32 normalized = ((unnormalized - mean) / (std_dev + eps_)) * w + b;
            acv_data[i + ctx_offset] = fpcvt_from_fp32<Float16>(normalized);
        }
    }
    else
    {
        acv_cached_ = true;

        for (int ctx_i = 0; ctx_i < n_ctx; ctx_i++)
        {
            const int ctx_offset = ctx_i * n_embed;

            // Mean calculation.
            Float32 mean_accum = 0.0f;
            for (int i = 0; i < n_embed; i++)
                mean_accum += fpcvt_to_fp32<Float16>(inp_data[i + ctx_offset]);
            Float32 mean = mean_accum / (Float32)n_embed;

            // Standard deviation calculation.
            Float32 variance_accum = 0.0f;
            for (int i = 0; i < n_embed; i++) {
                Float32 x = fpcvt_to_fp32<Float16>(inp_data[i + ctx_offset]);
                variance_accum += (x - mean) * (x - mean);
            }
            Float32 std_dev = std::sqrt(variance_accum / (Float32)n_embed);

            // Normalization.
            for (int i = 0; i < n_embed; i++) {
                Float32 x = fpcvt_to_fp32<Float16>(inp_data[i + ctx_offset]);
                Float32 w = fpcvt_to_fp32<Float16>(weight_data[i]);
                Float32 b = fpcvt_to_fp32<Float16>(bias_data[i]);
                // Epsilon added to standard deviation prevents div by zero.
                Float32 normalized = ((x - mean) / (std_dev + eps_)) * w + b;
                acv_data[i + ctx_offset] = fpcvt_from_fp32<Float16>(normalized);
            }
        }
    }

    return acv_;
}


GELU::GELU(const Tensor& acv)
    : acv_{acv}
{
}


Tensor GELU::forward(const Tensor& inp)
{
    Timer timer{&time_ms_};

    GTEN_CHECK_NDIMS_EQUAL(inp.ndims(), 2);
    const int n_out = acv_.size(1);
    GTEN_CHECK_DIMSIZE_EQUAL(1, inp.size(1), n_out);

    return forward_impl(inp);
}

Tensor GELU::forward_impl(const Tensor& inp)
{
    // TODO: Replace with tables.
    const int n_ctx = inp.size(0);
    const int n_out = acv_.size(1);

    acv_.resize({n_ctx, n_out});
    Float16* acv_data = acv_.data_ptr<Float16>();
    const Float16* inp_data = inp.data_ptr<Float16>();

    const int ne = inp.numel();
    if (acv_cached_) {
        const int start_i = (n_ctx - 1) * n_out;
        Float32 x;
        for (int i = start_i; i < ne; ++i) {
            x = fpcvt_to_fp32<Float16>(inp_data[i]);
            Float32 res = 0.5 * x 
                              * (1.0f + std::tanh(std::sqrt(2.0f / 3.141592653589793f)
                              * (x + 0.044715f * std::pow(x, 3.0f))));
            acv_data[i] = fpcvt_from_fp32<Float16>(res);
        }
    }
    else {
        acv_cached_ = true;

        Float32 x;
        for (int i = 0; i < ne; ++i) {
            x = fpcvt_to_fp32<Float16>(inp_data[i]);
            Float32 res = 0.5 * x 
                              * (1.0f + std::tanh(std::sqrt(2.0f / 3.141592653589793f)
                              * (x + 0.044715f * std::pow(x, 3.0f))));
            acv_data[i] = fpcvt_from_fp32<Float16>(res);
        }
    }
    return acv_;
}

Residual::Residual(const Tensor& acv)
    : acv_{acv}
{
}

Tensor Residual::forward(const Tensor& inp0, const Tensor& inp1)
{
    Timer timer{&time_ms_};

    GTEN_CHECK_DTYPE_EQUAL(inp0.dtype(), inp1.dtype());
    GTEN_CHECK_NDIMS_EQUAL(inp0.ndims(), 2);
    GTEN_CHECK_NDIMS_EQUAL(inp1.ndims(), 2);
    // TODO: Check shape inp1 == inp0

    return forward_impl(inp0, inp1);
}

Tensor Residual::forward_impl(const Tensor& inp0, const Tensor& inp1)
{
    const int n_ctx = inp0.size(0);
    const int n_embed = inp0.size(1);

    acv_.resize({n_ctx, n_embed});
    Float16* acv_data = acv_.data_ptr<Float16>();
    const Float16* inp0_data = inp0.data_ptr<Float16>();
    const Float16* inp1_data = inp1.data_ptr<Float16>();

    if (acv_cached_) {
        uint32_t n_iter = n_embed;
        uint32_t offset = inp0.numel() - n_embed;
        for (uint32_t i = 0; i < n_iter; i += 8) {
            Vec_f32x8 x0 = vec_f32x8_load(inp0_data + offset + i);
            Vec_f32x8 x1 = vec_f32x8_load(inp1_data + offset + i);
            Vec_f32x8 x_sum = vec_f32x8_add(x0, x1);
            vec_f32x8_store(x_sum, acv_data + offset + i);
        }
    }
    else {
        acv_cached_ = true;

        int n_iter = inp0.numel();
        for (int i = 0; i < n_iter; i += 8) {
            Vec_f32x8 x0 = vec_f32x8_load(inp0_data + i);
            Vec_f32x8 x1 = vec_f32x8_load(inp1_data + i);
            Vec_f32x8 x_sum = vec_f32x8_add(x0, x1);
            vec_f32x8_store(x_sum, acv_data + i);
        }
    }

    return acv_;
}

Linear::Linear(const Tensor &weight, const Tensor &bias,const Tensor& acv)
    : weight_{weight}, bias_{bias}, acv_{acv}
{
    GTEN_CHECK_NDIMS_EQUAL(weight_.ndims(), 2);
    GTEN_CHECK_NDIMS_EQUAL(bias_.ndims(), 1);
}


Tensor Linear::forward(const Tensor &inp)
{
    Timer timer{&time_ms_};

    GTEN_CHECK_NDIMS_EQUAL(inp.ndims(), 2);
    GTEN_CHECK_DIMSIZE_EQUAL(1, inp.size(1), weight_.size(1));

    return forward_impl(inp);
}

Tensor Linear::forward_impl(const Tensor& inp)
{
    const int n_ctx = inp.size(0);
    const int n_embed = inp.size(1);
    const int n_out = weight_.size(0);
    
    acv_.resize({n_ctx, n_out});
    Float16* acv_data = acv_.data_ptr<Float16>();
    const Float16* weight_data = weight_.data_ptr<Float16>();
    const Float16* bias_data = bias_.data_ptr<Float16>();
    const Float16* inp_data = inp.data_ptr<Float16>();

    if (acv_cached_)
    {
        const int ctx_i = n_ctx - 1;
        for (int out_i = 0; out_i < n_out; out_i++) {
            Vec_f32x8 dot_accum = vec_f32x8_setzero();
            for (int i = 0; i < n_embed; i += 8) {
                Vec_f32x8 x0 = vec_f32x8_load(inp_data + (ctx_i * n_embed + i));
                Vec_f32x8 x1 = vec_f32x8_load(weight_data + (out_i * n_embed + i));
                dot_accum = vec_f32x8_fma(x0, x1, dot_accum);
            }
            Float32 bias = fpcvt_to_fp32<Float16>(bias_data[out_i]);
            Float32 res =  vec_f32x8_sum(dot_accum) + bias;
            acv_data[ctx_i * n_out + out_i] = fpcvt_from_fp32<Float16>(res);
        }
    }
    else
    {
        acv_cached_ = true;
        for (int ctx_i = 0; ctx_i < n_ctx; ctx_i++) {
            for (int out_i = 0; out_i < n_out; out_i++) {
                Vec_f32x8 dot_accum = vec_f32x8_setzero();
                for (int i = 0; i < n_embed; i += 8) {
                    Vec_f32x8 x0 = vec_f32x8_load(inp_data + (ctx_i * n_embed + i));
                    Vec_f32x8 x1 = vec_f32x8_load(weight_data + (out_i * n_embed + i));
                    dot_accum = vec_f32x8_fma(x0, x1, dot_accum);
                }
                Float32 bias = fpcvt_to_fp32<Float16>(bias_data[out_i]);
                Float32 res =  vec_f32x8_sum(dot_accum) + bias;
                acv_data[ctx_i * n_out + out_i] = fpcvt_from_fp32<Float16>(res);
            }
        }
    }

    return acv_;
}

MultiHeadSelfAttn::MultiHeadSelfAttn(const Linear& query, const Linear& key,
                                     const Linear& value, const Linear& out_proj,
                                     const Tensor& qk_acv, const Tensor& qkv_acv,
                                     int n_head)
    : query_{query}, key_{key}, value_{value}, qkv_proj_{out_proj}, qk_acv_{qk_acv},
      qkv_acv_{qkv_acv}, n_head_{n_head}
{
    if (qk_acv_.dtype() == kFloat16) {
        Float16 *qk_cache_data = qk_acv_.data_ptr<Float16>();
        const int ne = qk_acv_.numel();
        const Float16 inf = fp32_to_fp16(-INFINITY);
        for (int i = 0; i < ne; i++)
            qk_cache_data[i] = inf;
    }
    else {
        Float32 *qk_cache_data = qk_acv_.data_ptr<Float32>();
        const int ne = qk_acv_.numel();
        for (int i = 0; i < ne; i++)
            qk_cache_data[i] = -INFINITY;
    }
}

Tensor MultiHeadSelfAttn::forward(const Tensor &inp)
{
    GTEN_CHECK_NDIMS_EQUAL(inp.ndims(), 2);
    GTEN_CHECK_DIMSIZE_EQUAL(1, inp.size(1), qkv_acv_.size(1));

    const int n_ctx = inp.size(0);
    const int n_embed = inp.size(1);

    Tensor q = query_.forward(inp);
    Tensor k = key_.forward(inp);
    Tensor v = value_.forward(inp);

    const Tensor qkv = qkv_attn(q, k, v);
    const Tensor out = qkv_proj_.forward(qkv);
    return out;
}

Tensor MultiHeadSelfAttn::qkv_attn(const Tensor &q, const Tensor &k, const Tensor &v)
{
    Timer timer(&time_attn_ms_);

    const int n_ctx = q.size(0);
    const int n_embed = q.size(1);
    const int d_head = n_embed / n_head_;

    const Float16* q_data = q.data_ptr<Float16>();
    const Float16* k_data = k.data_ptr<Float16>();

    qk_acv_.resize({n_head_, n_ctx, n_ctx});
    Float16* qk_data = qk_acv_.data_ptr<Float16>();

    qkv_acv_.resize({n_ctx, n_embed});
    Float16* qkv_data = qkv_acv_.data_ptr<Float16>();
    const Float16* v_data = v.data_ptr<Float16>();

    Float32 scale_factor = 1.0f / std::sqrt((Float32)d_head);
    if (qkv_cached_)
    {
        // SCALED DOT-PRODUCT: (Q @ K) * scale.
        for (int head = 0; head < n_head_; head++)
        {
            const int q_row = n_ctx - 1;
            for (int k_col = 0; k_col < n_ctx; k_col++) {
                Vec_f32x8 dot_accum = vec_f32x8_setzero();
                for (int i = 0; i < d_head; i += 8) {
                    int q_i = head * d_head + q_row * n_embed + i;
                    int k_i = head * d_head + k_col * n_embed + i;
                    Vec_f32x8 qw = vec_f32x8_load(q_data + q_i);
                    Vec_f32x8 kw = vec_f32x8_load(k_data + k_i);
                    dot_accum = vec_f32x8_add(vec_f32x8_mul(qw, kw), dot_accum);
                }
                Float32 dot_prod = vec_f32x8_sum(dot_accum) * scale_factor;
                int qk_data_i = head * n_ctx * n_ctx + q_row * n_ctx + k_col;
                qk_data[qk_data_i] = fpcvt_from_fp32<Float16>(dot_prod);
            }
        }

        // SOFTMAX
        for (int head = 0; head < n_head_; head++)
        {
            const int row = n_ctx - 1;
            const int base_i = head * n_ctx * n_ctx + row * n_ctx;

            Float32 max = -INFINITY;
            for (int i = 0; i < n_ctx; i++) {
                Float32 val = fpcvt_to_fp32<Float16>(qk_data[base_i + i]);
                if (val > max)
                    max = val;
            }

            Float32 sum_exp = 0;
            for (int i = 0; i < n_ctx; i++) {
                int qk_i = base_i + i;
                Float32 val = fpcvt_to_fp32<Float16>(qk_data[qk_i]);
                qk_data[qk_i] = fpcvt_from_fp32<Float16>(std::exp(val - max));
                sum_exp += fpcvt_to_fp32<Float16>(qk_data[qk_i]);
            }

            for (int i = 0; i < n_ctx; i++) {
                Float32 qkw = fpcvt_to_fp32<Float16>(qk_data[base_i + i]);
                qk_data[base_i + i] = fpcvt_from_fp32<Float16>(qkw / sum_exp);
            }
        }

        // ATTENTION: QK @ V
        for (int head = 0; head < n_head_; head++)
        {
            const int qk_row = n_ctx - 1;
            for (int v_col = 0; v_col < d_head; v_col++) {
                Float32 dot_prod = 0;
                for (int i = 0; i < n_ctx; i++) {
                    int qk_i = head * n_ctx * n_ctx + qk_row * n_ctx + i;
                    int v_i = head * d_head + i * n_embed + v_col;
                    Float32 qkw = fpcvt_to_fp32<Float16>(qk_data[qk_i]);
                    Float32 vw = fpcvt_to_fp32<Float16>(v_data[v_i]);
                    dot_prod += qkw * vw;
                }
                int qkv_data_i = head * d_head + qk_row * n_embed + v_col;
                qkv_data[qkv_data_i] = fpcvt_from_fp32<Float16>(dot_prod);
            }
        }
    }
    else
    {
        qkv_cached_ = true;

        // SCALED DOT-PRODUCT: (Q @ K) * scale.
        for (int head = 0; head < n_head_; head++)
        {
            for (int q_row = 0; q_row < n_ctx; q_row++)
            {
                // `non_masked_prods` represents the number of dot products that will
                // not be masked and therefore must be computed. This allows us to skip
                // unecessary dot products.
                int non_masked_prods = q_row + 1;
                for (int k_col = 0; k_col < non_masked_prods; k_col++) {
                    Float32 dot_accum = 0.0f;
                    for (int i = 0; i < d_head; i++) {
                        int q_i = head * d_head + q_row * n_embed + i;
                        int k_i = head * d_head + k_col * n_embed + i;
                        Float32 qw = fpcvt_to_fp32<Float16>(q_data[q_i]);
                        Float32 kw = fpcvt_to_fp32<Float16>(k_data[k_i]);
                        dot_accum += qw * kw * scale_factor;
                    }
                    int qk_data_i = head * n_ctx * n_ctx + q_row * n_ctx + k_col;
                    qk_data[qk_data_i] = fpcvt_from_fp32<Float16>(dot_accum);
                }

            }
        }

        // SOFTMAX
        int n_rows = n_head_ * n_ctx;
        for (int row = 0; row < n_rows; row++)
        {
            Float32 max = -INFINITY;
            for (int i = 0; i < n_ctx; i++) {
                Float32 x = fpcvt_to_fp32<Float16>(qk_data[row * n_ctx + i]);
                if (x > max)
                    max = x;
            }

            Float32 sum_exp = 0;
            for (int i = 0; i < n_ctx; i++) {
                int qk_i = row * n_ctx + i;
                Float32 x = fpcvt_to_fp32<Float16>(qk_data[qk_i]);
                qk_data[qk_i] = fpcvt_from_fp32<Float16>(std::exp(x - max));
                sum_exp += fpcvt_to_fp32<Float16>(qk_data[qk_i]);
            }

            for (int i = 0; i < n_ctx; i++) {
                Float32 qkw =  fpcvt_to_fp32<Float16>(qk_data[row * n_ctx + i]);
                qk_data[row * n_ctx + i] = fpcvt_from_fp32<Float16>(qkw / sum_exp);
            }
        }

        // ATTENTION: QK @ V
        for (int head = 0; head < n_head_; head++)
        {
            for (int qk_row = 0; qk_row < n_ctx; qk_row++){
                for (int v_col = 0; v_col < d_head; v_col++) {
                    Float32 dot_prod = 0;
                    for (int i = 0; i < n_ctx; i++) {
                        int qk_i = head * n_ctx * n_ctx + qk_row * n_ctx + i;
                        int v_i = head * d_head + i * n_embed + v_col;
                        Float32 qkw =fpcvt_to_fp32<Float16>(qk_data[qk_i]);
                        Float32 vw = fpcvt_to_fp32<Float16>(v_data[v_i]);
                        dot_prod += qkw * vw;
                    }
                    int qkv_data_i = head * d_head + qk_row * n_embed + v_col;
                    qkv_data[qkv_data_i] = fpcvt_from_fp32<Float16>(dot_prod);
                }
            }
        }
    }

    return qkv_acv_;
}

ResidualAttentionBlock::ResidualAttentionBlock(const MultiHeadSelfAttn& attn,
                                               const LayerNorm& ln_1,
                                               const Linear& mlp_fc,
                                               const Linear& mlp_proj,
                                               const LayerNorm& ln_2,
                                               const GELU& gelu,
                                               const Residual& inp_res,
                                               const Residual& attn_res,
                                               int32_t max_ctx, int32_t n_embed)
    : attn_{attn}, ln_1_{ln_1}, mlp_fc_{mlp_fc}, mlp_proj_{mlp_proj}, ln_2_{ln_2},
      gelu_{gelu}, inp_res_{inp_res}, attn_res_{attn_res}
{
}

Tensor ResidualAttentionBlock::forward(const Tensor &inp)
{
    Tensor attn = inp_res_.forward(inp, attn_.forward(ln_1_.forward(inp)));
    Tensor out = attn_res_.forward(attn,
        mlp_proj_.forward(gelu_.forward(mlp_fc_.forward(ln_2_.forward(attn)))));
    return out;
}

} // namespace gten
