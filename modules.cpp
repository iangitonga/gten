#include "modules.h"

#include <iostream>
#include <cmath>

#if defined(__AVX__)
#define USE_AVX 1
#else
#define USE_AVX 0
#endif

#if USE_AVX
#include <immintrin.h>
#endif

namespace gten
{

Embedding::Embedding(const Tensor &weight, uint32_t max_ctx)
    : weight_(weight), max_ctx_(max_ctx)
{
    GTEN_CHECK(weight.dtype() == Dtype::Float32, "Embedding weight must be of Float32 dtype.");
    GTEN_CHECK(weight.is_2d(), "Embedding weight must be a 2-d tensor. The provided weight has %ld dims.", weight.ndims());

    n_vocab_ = weight.size(0);
    n_embed_ = weight.size(1);
    emb_out_ = Tensor({max_ctx_, n_embed_});
    proj_out_ = Tensor({1, n_vocab_});
}

Tensor Embedding::forward(const Tensor &inp)
{
    GTEN_CHECK(inp.dtype() == Dtype::Int32, "Input tokens tensor is not of dtype Int32.");
    GTEN_CHECK(inp.is_1d(), "Input tokens tensor is not 1-d. It has %ld dims.", inp.ndims());
    GTEN_CHECK(inp.numel() <= max_ctx_, "Input tokens of length %d exceed max context size of %d.", inp.numel(), max_ctx_);

    Timer timer(&time_embed_ms_);

    const int32_t *inp_data = inp.data_ptr<int32_t>();
    const float *weight_data = weight_.data_ptr<float>();
    emb_out_.reshape({inp.numel(), n_embed_});
    float *out_data = emb_out_.data_ptr<float>();

    if (emb_out_cached_)
    {
        const uint32_t token_i = inp.numel() - 1;
        const uint32_t emb_i = inp_data[token_i] * n_embed_;
        void * src = (void *)(weight_data + emb_i);
        void * dest = (void *)(out_data + token_i * n_embed_);
        std::memcpy(dest, src, n_embed_ * weight_.bytes_per_item());
    }
    else
    {
        emb_out_cached_ = true;
        for (uint32_t token_i = 0; token_i < inp.numel(); token_i++)
        {
            int32_t emb_i = inp_data[token_i] * n_embed_;
            void * src = (void *)(weight_data + emb_i);
            void * dest = (void *)(out_data + token_i * n_embed_);
            std::memcpy(dest, src, n_embed_ * weight_.bytes_per_item());
        }
    }

    return emb_out_;
}


Tensor Embedding::forward_project(const Tensor &inp)
{
    GTEN_CHECK(inp.dtype() == Dtype::Float32, "Input tensor is not of dtype FLoat32.");
    GTEN_CHECK(inp.is_2d(), "Input tensor is not 2-d. It has %ld dims.", inp.ndims());
    GTEN_CHECK(inp.size(1) == n_embed_, "Input is expected to have size %d in the second dim, it has %d.", inp.size(1), n_embed_);

    Timer timer(&time_project_ms_);

    const uint32_t n_ctx = inp.shape()[0];
    const uint32_t n_embed = inp.shape()[1];
    const uint32_t n_vocab = n_vocab_;

    const float *inp_data = inp.data_ptr<float>();
    const float *weight_data = weight_.data_ptr<float>();
    float *out_data = proj_out_.data_ptr<float>();

    const uint32_t ctx_i = n_ctx - 1;

#if USE_AVX
    for (uint32_t emb_i = 0; emb_i < n_vocab; emb_i++)
    {
        float dot_prod = 0.0f;
        float *res;
        __m256 dot_accum = _mm256_setzero_ps();

        for (uint32_t el = 0; el < n_embed; el += 8)
        {
            __m256 iv = _mm256_loadu_ps(inp_data + ctx_i * n_embed + el);
            __m256 wv = _mm256_loadu_ps(weight_data + emb_i * n_embed + el);
            __m256 rv = _mm256_mul_ps(iv, wv);

            dot_accum = _mm256_add_ps(dot_accum, rv);
        }

        res = (float *)(&dot_accum);
        out_data[emb_i] = res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7];
    }    
#else
    for (uint32_t emb_i = 0; emb_i < n_vocab; emb_i++)
    {
        float dot_prod = 0.0f;
        for (uint32_t el = 0; el < n_embed; el++)
            dot_prod += inp_data[ctx_i * n_embed + el] * weight_data[emb_i * n_embed + el];
        out_data[emb_i] = dot_prod;
    }
#endif

    return proj_out_;
}

PosEmbedding::PosEmbedding(const Tensor &weight, uint32_t max_ctx)
    : weight_(weight), max_ctx_(max_ctx)
{
    GTEN_CHECK(weight.dtype() == Dtype::Float32, "The given positional embedding weight is not of Float32 dtype.");
    GTEN_CHECK(weight.is_2d(), "The given positional embedding weight is not a 2-d tensor. It has has %ld dims.", weight.ndims());

    n_embed_ = weight.size(1);
    out_ = Tensor({max_ctx, n_embed_});
}

Tensor PosEmbedding::forward(uint32_t n_ctx)
{
    GTEN_CHECK(n_ctx > 0, "Given context length is out of range.");
    GTEN_CHECK(n_ctx <= max_ctx_, "Given context length is out of range.");

    Timer timer(&time_ms_);

    const float *weight_data = weight_.data_ptr<float>();
    // TODO: Instead of this, we could return a tensor sharing data with table data.
    out_.reshape({n_ctx, n_embed_});
    float *out_data = out_.data_ptr<float>();

    if (out_cached_) {
        const uint32_t data_offset = (n_ctx - 1) * n_embed_;
        size_t bytes_to_copy = n_embed_ * weight_.bytes_per_item();
        std::memcpy(out_data + data_offset, weight_data + data_offset, bytes_to_copy);
    }
    else {
        out_cached_ = true;
        size_t bytes_to_copy = n_ctx * n_embed_ * weight_.bytes_per_item();
        std::memcpy(out_data, weight_data, bytes_to_copy);
    }
    return out_;
}

LayerNorm::LayerNorm(const Tensor &weight, const Tensor &bias, uint32_t max_ctx)
    : weight_(weight), bias_(bias), max_ctx_(max_ctx)
{
    GTEN_CHECK(weight.dtype() == Dtype::Float32, "The given layernorm weight is not of Float32 dtype.");
    GTEN_CHECK(bias.dtype() == Dtype::Float32, "The given layernorm bias is not of Float32 dtype.");
    GTEN_CHECK(weight.is_1d(), "The given layernorm weight is not a 1-d tensor. It has has %ld dims.", weight.ndims());
    GTEN_CHECK(bias.is_1d(), "The given layernorm bias is not a 1-d tensor. It has has %ld dims.", bias.ndims());

    n_embed_ = weight.size(0);
    out_ = Tensor({max_ctx, n_embed_}, Dtype::Float32);
}

Tensor LayerNorm::forward(const Tensor &inp)
{
    GTEN_CHECK(inp.is_2d(), "Layernorm input tensor is not a 2d tensor.");
    GTEN_CHECK(inp.size(0) <= max_ctx_, "Input's context size of %d, exceeds max context size of %d.", inp.size(0), max_ctx_);
    GTEN_CHECK(inp.size(1) == n_embed_, "Input is expected to have size %d in the second dim, it has %d.", inp.size(1), n_embed_);
    
    Timer timer(&time_ms_);

    out_.reshape(inp.shape());
    float *out_data = out_.data_ptr<float>();

    const float *inp_data = inp.data_ptr<float>();
    const float *weight_data = weight_.data_ptr<float>();
    const float *bias_data = bias_.data_ptr<float>();

    const uint32_t n_ctx = inp.shape()[0];  //n_ctx

    if (out_cached_)
    {
        const uint32_t ctx_offset = (n_ctx - 1) * n_embed_;
        // Mean calculation.
        float mean_accum = 0.0f;
        for (uint32_t el = 0; el < n_embed_; el++)
            mean_accum += inp_data[el + ctx_offset];
        float mean = mean_accum / (float)n_embed_;

        // Standard deviation calculation.
        float variance_accum = 0.0f;
        for (uint32_t el = 0; el < n_embed_; el++)
            variance_accum += std::pow(inp_data[el + ctx_offset] - mean, 2);
        float std_dev = std::sqrt(variance_accum / (float)n_embed_);

        // Normalization.
        for (uint32_t el = 0; el < n_embed_; el++)
        {
            float unnormalized = inp_data[el + ctx_offset];
            // Epsilon added to standard deviation prevents div by zero.
            float normalized = ((unnormalized - mean) / (std_dev + eps_)) * weight_data[el] + bias_data[el];
            out_data[el + ctx_offset] = normalized;
        }
    }
    else
    {
        out_cached_ = true;
        for (uint32_t ctx_i = 0; ctx_i < n_ctx; ctx_i++)
        {
            const uint32_t ctx_offset = ctx_i * n_embed_;

            // Mean calculation.
            float mean_accum = 0.0f;
            for (uint32_t el_idx = 0; el_idx < n_embed_; el_idx++)
                mean_accum += inp_data[el_idx + ctx_offset];
            float mean = mean_accum / (float)n_embed_;

            // Standard deviation calculation.
            float variance_accum = 0.0f;
            for (uint32_t el_idx = 0; el_idx < n_embed_; el_idx++)
                variance_accum += std::pow(inp_data[el_idx + ctx_offset] - mean, 2);
            float std_dev = std::sqrt(variance_accum / (float)n_embed_);

            // Normalization.
            for (uint32_t el = 0; el < n_embed_; el++)
            {
                float unnormalized = inp_data[el + ctx_offset];
                // Epsilon added to standard deviation prevents div by zero.
                float normalized = ((unnormalized - mean) / (std_dev + eps_)) * weight_data[el] + bias_data[el];
                out_data[el + ctx_offset] = normalized;
            }

        }
    }
    return out_;
}

Linear::Linear(const Tensor &weight, const Tensor &bias, uint32_t max_ctx)
    : weight_(weight), bias_(bias)
{
    // color, line, fn
    GTEN_CHECK(weight.dtype() == Dtype::Float32, "The given linear weight is not of Float32 dtype.");
    GTEN_CHECK(bias.dtype() == Dtype::Float32, "The given linear bias is not of Float32 dtype.");
    GTEN_CHECK(weight.is_2d(), "The given linear weight is not a 2-d tensor. It has has %ld dims.", weight.ndims());
    GTEN_CHECK(bias.is_1d(), "The given linear bias is not a 1-d tensor. It has has %ld dims.", bias.ndims());
    
    out_ = Tensor({max_ctx, weight_.size(0)});
}

Tensor Linear::forward(const Tensor &inp)
{
    GTEN_CHECK(inp.is_2d(), "Linear input tensor is not a 2d tensor.");

    Timer timer(&time_ms_);

    const uint32_t n_ctx = inp.shape()[0];
    const uint32_t n_embed = inp.shape()[1];
    const uint32_t n_out = weight_.shape()[0];

    out_.reshape({n_ctx, n_out});
    float *out_data = out_.data_ptr<float>();
    const float *weight_data = weight_.data_ptr<float>();
    const float *bias_data = bias_.data_ptr<float>();
    const float *inp_data = inp.data_ptr<float>();

#if USE_AVX
    if (out_cached_)
    {
        const uint32_t ctx_i = n_ctx - 1;

        for (uint32_t out_i = 0; out_i < n_out; out_i++)
        {
            float *res;
            __m256 dot_accum = _mm256_setzero_ps();

            for (uint32_t el = 0; el < n_embed; el += 8) {
                __m256 iv = _mm256_loadu_ps(inp_data + (ctx_i * n_embed + el));
                __m256 wv = _mm256_loadu_ps(weight_data + (out_i * n_embed + el));
                __m256 rv = _mm256_mul_ps(iv, wv);

                dot_accum = _mm256_add_ps(dot_accum, rv);
            }
            res = (float *)(&dot_accum);
            out_data[ctx_i * n_out + out_i] = res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7] + bias_data[out_i];
        }
    }
    else
    {
        out_cached_ = true;
        for (uint32_t ctx_i = 0; ctx_i < n_ctx; ctx_i++)
        {
            for (uint32_t out_i = 0; out_i < n_out; out_i++)
            {
                float dot_prod = 0.0f;
                float *res;
                __m256 dot_accum = _mm256_setzero_ps();
                for (uint32_t el = 0; el < n_embed; el += 8)
                {
                    __m256 iv = _mm256_loadu_ps(inp_data + (ctx_i * n_embed + el));
                    __m256 wv = _mm256_loadu_ps(weight_data + (out_i * n_embed + el));
                    __m256 rv = _mm256_mul_ps(iv, wv);
                    dot_accum = _mm256_add_ps(dot_accum, rv);
                }
                res = (float *)(&dot_accum);
                out_data[ctx_i * n_out + out_i] = res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7] + bias_data[out_i];
            }
        }
    }
#else
    if (out_cached_)
    {
        const uint32_t ctx_i = n_ctx - 1;
        for (uint32_t out_i = 0; out_i < n_out; out_i++)
        {
            float dot_prod = 0.0f;
            for (uint32_t el = 0; el < n_embed; el++)
                dot_prod += inp_data[ctx_i * n_embed + el] * weight_data[out_i * n_embed + el];
            out_data[ctx_i * n_out + out_i] = dot_prod + bias_data[out_i];
        }
    }
    else
    {
        out_cached_ = true;
        for (uint32_t ctx_i = 0; ctx_i < n_ctx; ctx_i++)
        {
            for (uint32_t out_i = 0; out_i < n_out; out_i++)
            {
                float dot_prod = 0.0f;
                for (uint32_t el = 0; el < n_embed; el++)
                    dot_prod += inp_data[ctx_i * n_embed + el] * weight_data[out_i * n_embed + el];
                out_data[ctx_i * n_out + out_i] = dot_prod + bias_data[out_i];
            }
        }
    }
#endif

    return out_;
}

Residual::Residual(uint32_t max_ctx, uint32_t n_embed)
{
    out_ = Tensor({max_ctx, n_embed});
}

Tensor Residual::forward(const Tensor &inp0, const Tensor &inp1)
{
    GTEN_CHECK(inp0.numel() == inp1.numel(), "Addition of diff length tensors not allowed.");

    Timer timer(&time_ms_);

    const uint32_t n_ctx = inp0.shape()[0];
    const uint32_t n_embed = inp0.shape()[1];

    out_.reshape({n_ctx, n_embed});
    float *out_data = out_.data_ptr<float>();
    const float *inp0_data = inp0.data_ptr<float>();
    const float *inp1_data = inp1.data_ptr<float>();

#if USE_AVX
    if (out_cached_)
    {
        uint32_t n_iter = n_embed;
        uint32_t offset = inp0.numel() - n_embed;

        for (uint32_t i = 0; i < n_iter; i += 8) {
            __m256 v0 = _mm256_loadu_ps(inp0_data + offset + i);
            __m256 v1 = _mm256_loadu_ps(inp1_data + offset + i);
            __m256 rv = _mm256_add_ps(v0, v1);
            _mm256_storeu_ps(out_data + offset + i, rv);
        }
    }
    else {
        out_cached_ = true;
        uint32_t n_iter = inp0.numel();

        for (uint32_t i = 0; i < n_iter; i += 8) {
            __m256 v0 = _mm256_loadu_ps(inp0_data + i);
            __m256 v1 = _mm256_loadu_ps(inp1_data + i);
            __m256 rv = _mm256_add_ps(v0, v1);
            _mm256_storeu_ps(out_data + i, rv);
        }
    }
#else
if (out_cached_)
    {
        uint32_t n_iter = n_embed;
        uint32_t offset = inp0.numel() - n_embed;
        for (uint32_t i = 0; i < n_iter; i++)
            out_data[i + offset] = inp0_data[i + offset] + inp1_data[i + offset];
    }
    else {
        out_cached_ = true;
        uint32_t n_iter = inp0.numel();
        for (uint32_t i = 0; i < n_iter; i++)
            out_data[i] = inp0_data[i] + inp1_data[i];
    }
#endif
    
    return out_;
}

GELU::GELU(uint32_t max_ctx, uint32_t n_out)
    : n_out_(n_out)
{
    out_ = Tensor({max_ctx, n_out_}, Dtype::Float32);
}

Tensor GELU::forward(const Tensor &inp)
{
    Timer timer(&time_ms_);

    const uint32_t n_ctx = inp.shape()[0];
    const float *inp_data = inp.data_ptr<float>();
    out_.reshape({n_ctx, n_out_});
    float *out_data = out_.data_ptr<float>();

    const uint32_t ne = inp.numel();
    if (out_cached_)
    {
        const uint32_t start_i = (n_ctx - 1) * n_out_;
        float x;
        for (int i = start_i; i < ne; ++i) {
            x = inp_data[i];
            out_data[i] = 0.5 * x * (1.0f + std::tanh(std::sqrt(2.0f / 3.141592653589793f)
                                  * (x + 0.044715f * std::pow(x, 3.0f))));
        }
    }
    else
    {
        out_cached_ = true;
        float x;
        for (int i = 0; i < ne; ++i) {
            x = inp_data[i];
            out_data[i] = 0.5 * x * (1.0f + std::tanh(std::sqrt(2.0f / 3.141592653589793f)
                                  * (x + 0.044715f * std::pow(x, 3.0f))));
        }
    }
    return out_;
}

MultiHeadSelfAttn::MultiHeadSelfAttn(const Linear &query,
                                    const Linear &key,
                                    const Linear &value,
                                    const Linear &out_proj,
                                    uint32_t max_ctx,
                                    uint32_t n_embed,
                                    uint32_t n_head)
    : query_(query), key_(key), value_(value), out_proj_(out_proj), n_head_(n_head)
{
    qk_cache_ = Tensor({n_head, max_ctx, max_ctx});
    float *qk_cache_data = qk_cache_.data_ptr<float>();
    for (uint32_t i = 0; i < qk_cache_.numel(); i++)
        qk_cache_data[i] = -INFINITY;
    qkv_cache_ = Tensor({max_ctx, n_embed});
}

Tensor MultiHeadSelfAttn::forward(const Tensor &inp)
{
    const uint32_t n_ctx = inp.size(0);
    const uint32_t n_embed = inp.size(1);

    Tensor q = query_.forward(inp);
    Tensor k = key_.forward(inp);
    Tensor v = value_.forward(inp);

    const Tensor qkv = qkv_attn(q, k, v);
    const Tensor out = out_proj_.forward(qkv);
    return out;
}

Tensor MultiHeadSelfAttn::qkv_attn(const Tensor &q, const Tensor &k, const Tensor &v)
{
    Timer timer(&time_attn_ms_);

    const uint32_t n_ctx = q.size(0);
    const uint32_t n_embed = q.size(1);
    const uint32_t d_head = n_embed / n_head_;

    const float *q_data = q.data_ptr<float>();
    const float *k_data = k.data_ptr<float>();

    qk_cache_.reshape({n_head_, n_ctx, n_ctx});
    float *qk_data = qk_cache_.data_ptr<float>();

    qkv_cache_.reshape({n_ctx, n_embed});
    float *qkv_data = qkv_cache_.data_ptr<float>();
    const float *v_data = v.data_ptr<float>();

    float scale_factor = 1.0f / std::sqrt((float)d_head);
    if (qkv_cached_)
    {
        // SCALED DOT-PRODUCT: (Q @ K) * scale.
        for (uint32_t head = 0; head < n_head_; head++)
        {
            const uint32_t q_row = n_ctx - 1;
            for (uint32_t k_col = 0; k_col < n_ctx; k_col++)
            {
                float dot_prod = 0.0f;
                for (uint32_t el = 0; el < d_head; el++)
                {
                    uint32_t q_idx = head * d_head + q_row * n_embed + el;
                    uint32_t k_idx = head * d_head + k_col * n_embed + el;
                    dot_prod += q_data[q_idx] * k_data[k_idx] * scale_factor;
                }
                qk_data[head * n_ctx * n_ctx + q_row * n_ctx + k_col] = dot_prod;
            }
        }

        // SOFTMAX
        for (uint32_t head = 0; head < n_head_; head++)
        {
            const uint32_t row = n_ctx - 1;
            const uint32_t base_i = head * n_ctx * n_ctx + row * n_ctx;
            float max = -INFINITY;
            for (uint32_t el = 0; el < n_ctx; el++)
                if (qk_data[base_i + el] > max)
                    max = qk_data[base_i + el];

            float sum_exp = 0;
            for (uint32_t el = 0; el < n_ctx; el++)
            {
                uint32_t idx = base_i + el;
                qk_data[idx] = std::exp(qk_data[idx] - max);
                sum_exp += qk_data[idx];
            }

            for (uint32_t el = 0; el < n_ctx; el++)
                qk_data[base_i + el] /= sum_exp;
            
        }

        // ATTENTION: QK @ V
        for (uint32_t head = 0; head < n_head_; head++)
        {
            const uint32_t qk_row = n_ctx - 1;
            for (uint32_t v_col = 0; v_col < d_head; v_col++)
            {
                float dot_prod = 0;
                for (uint32_t el = 0; el < n_ctx; el++)
                {
                    uint32_t qk_idx = head * n_ctx * n_ctx + qk_row * n_ctx + el;
                    uint32_t v_idx = head * d_head + el * n_embed + v_col;
                    dot_prod += qk_data[qk_idx] * v_data[v_idx];
                }
                qkv_data[head * d_head + qk_row * n_embed + v_col] = dot_prod;
            }
        }
    }
    else
    {
        qkv_cached_ = true;

        // SCALED DOT-PRODUCT: (Q @ K) * scale.
        for (uint32_t head = 0; head < n_head_; head++)
        {
            for (uint32_t q_row = 0; q_row < n_ctx; q_row++)
            {
                // `non_masked_prods` represents the number of dot products that will not be masked
                // and therefore must be computed. This allows us to skip unecessary dot products.
                uint32_t non_masked_prods = q_row + 1;
                for (uint32_t k_col = 0; k_col < non_masked_prods; k_col++)
                {
                    float dot_prod = 0.0f;
                    for (uint32_t el = 0; el < d_head; el++)
                    {
                        uint32_t q_idx = head * d_head + q_row * n_embed + el;
                        uint32_t k_idx = head * d_head + k_col * n_embed + el;
                        dot_prod += q_data[q_idx] * k_data[k_idx] * scale_factor;
                    }
                    qk_data[head * n_ctx * n_ctx + q_row * n_ctx + k_col] = dot_prod;
                }

            }
        }

        // SOFTMAX
        uint32_t n_rows = n_head_ * n_ctx;
        for (uint32_t row = 0; row < n_rows; row++)
        {
            float max = -INFINITY;
            for (uint32_t el = 0; el < n_ctx; el++)
                if (qk_data[row * n_ctx + el] > max)
                    max = qk_data[row * n_ctx + el];

            float sum_exp = 0;
            for (uint32_t el = 0; el < n_ctx; el++)
            {
                uint32_t idx = row * n_ctx + el;
                qk_data[idx] = std::exp(qk_data[idx] - max);
                sum_exp += qk_data[idx];
            }

            for (uint32_t el = 0; el < n_ctx; el++)
                qk_data[row * n_ctx + el] /= sum_exp;
        }

        // ATTENTION: QK @ V
        for (uint32_t head = 0; head < n_head_; head++)
        {
            for (uint32_t qk_row = 0; qk_row < n_ctx; qk_row++)
            {
                for (uint32_t v_col = 0; v_col < d_head; v_col++)
                {
                    float dot_prod = 0;
                    for (uint32_t el = 0; el < n_ctx; el++)
                    {
                        uint32_t qk_idx = head * n_ctx * n_ctx + qk_row * n_ctx + el;
                        uint32_t v_idx = head * d_head + el * n_embed + v_col;
                        dot_prod += qk_data[qk_idx] * v_data[v_idx];
                    }
                    qkv_data[head * d_head + qk_row * n_embed + v_col] = dot_prod;
                }
            }
        }
    }

    return qkv_cache_;
}

ResidualAttentionBlock::ResidualAttentionBlock(const MultiHeadSelfAttn &attn,
                                               const LayerNorm &ln_1,
                                               const Linear &mlp_fc,
                                               const Linear &mlp_proj,
                                               const LayerNorm &ln_2,
                                               const GELU &gelu,
                                               uint32_t max_ctx,
                                               uint32_t n_embed)
    : attn_(attn), ln_1_(ln_1), mlp_fc_(mlp_fc), mlp_proj_(mlp_proj), ln_2_(ln_2), gelu_(gelu)
{
    inp_res_ = Residual(max_ctx, n_embed);
    attn_res_ = Residual(max_ctx, n_embed);
}

Tensor ResidualAttentionBlock::forward(const Tensor &inp)
{
    Tensor attn = inp_res_.forward(inp, attn_.forward(ln_1_.forward(inp)));
    Tensor out = attn_res_.forward(attn, mlp_proj_.forward(gelu_.forward(mlp_fc_.forward(ln_2_.forward(attn)))));
    return out;
}

} // namespace gten
