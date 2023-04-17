#include "modules.h"
#include "tokenizer.h"

#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>

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

// Any module forward call must not modify input tensor. i.e it must accept `const Tensor&`.

Embedding::Embedding(const Tensor &weight, const uint32_t n_vocab, const uint32_t max_ctx, const uint32_t n_embed)
    : weight_(weight), n_vocab_(n_vocab), n_embed_(n_embed)
{
    emb_out_ = Tensor({max_ctx, n_embed});
    proj_out_ = Tensor({1, n_vocab});
}

Tensor Embedding::forward(const Tensor &inp)
{
    GTEN_CHECK(inp.dtype() == Dtype::Int32, "Embedding input tensor is not of `int32` type.");
    GTEN_CHECK(inp.is_1d(), "Embedding input tensor is not a 1d tensor.");

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

PosEmbedding::PosEmbedding(const Tensor &weight, const uint32_t max_ctx, const uint32_t n_embed)
    : max_ctx_(max_ctx), n_embed_(n_embed), weight_(weight)
{
    out_ = Tensor({max_ctx, n_embed});
}

Tensor PosEmbedding::forward(const uint32_t n_ctx)
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

LayerNorm::LayerNorm(const Tensor &weight, const Tensor &bias, const uint32_t max_ctx, const uint32_t n_embed)
    : n_embed_(n_embed), weight_(weight), bias_(bias)
{
    out_ = Tensor({max_ctx, 768}, Dtype::Float32);
}

Tensor LayerNorm::forward(const Tensor &inp)
{
    GTEN_CHECK(inp.is_2d(), "Layernorm input tensor is not a 2d tensor.");
    
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

Linear::Linear(const Tensor &weight, const Tensor &bias, const uint32_t max_ctx)
    : weight_(weight), bias_(bias)
{
    out_ = Tensor({max_ctx, weight_.shape()[0]});
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

Residual::Residual(const uint32_t max_ctx, const uint32_t n_embed)
{
    out_ = Tensor({max_ctx, n_embed});
}

Tensor Residual::forward(const Tensor &inp0, const Tensor &inp1)
{
    GTEN_CHECK(inp0.numel() == inp1.numel(), "Addition of diff length tensors not allowed.");

    Timer timer(&time_ms_);

    // inp (n_ctx, n_embed)
    // in2 (n_ctx, n_embed)

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

GELU::GELU(const uint32_t max_ctx, const uint32_t n_out)
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
            out_data[i] = 0.5 * x * (1.0f + std::tanh(std::sqrt(2.0f / 3.141592653589793f) * (x + 0.044715f * std::pow(x, 3.0f))));
        }
    }
    else
    {
        out_cached_ = true;
        float x;
        for (int i = 0; i < ne; ++i) {
            x = inp_data[i];
            out_data[i] = 0.5 * x * (1.0f + std::tanh(std::sqrt(2.0f / 3.141592653589793f) * (x + 0.044715f * std::pow(x, 3.0f))));
        }
    }
    return out_;
}

MultiHeadSelfAttn::MultiHeadSelfAttn(const Linear &query, const Linear &key, const Linear &value, const Linear &out_proj,
                                    const uint32_t max_ctx, const uint32_t n_embed)
    : query_(query), key_(key), value_(value), out_proj_(out_proj)
{
    qk_cache_ = Tensor({12, max_ctx, max_ctx});
    float *qk_cache_data = qk_cache_.data_ptr<float>();
    for (uint32_t i = 0; i < qk_cache_.numel(); i++)
        qk_cache_data[i] = -INFINITY;
    qkv_cache_ = Tensor({max_ctx, n_embed});
}

Tensor MultiHeadSelfAttn::forward(const Tensor &inp)
{
    const uint32_t n_ctx = inp.shape()[0];
    const uint32_t n_embed = inp.shape()[1];

    Tensor q = query_.forward(inp);
    Tensor k = key_.forward(inp);
    Tensor v = value_.forward(inp);

    // TODO: Remove hardcoded.
    const uint32_t n_head = 12;
    const Tensor qkv = qkv_attn(q, k, v, n_head);
    const Tensor out = out_proj_.forward(qkv);
    return out;
}

Tensor MultiHeadSelfAttn::qkv_attn(const Tensor &q, const Tensor &k, const Tensor &v, const uint32_t n_head)
{
    Timer timer(&time_attn_ms_);

    // self attn.
    // TODO API: q.shape(0), q.fill(arr)
    const uint32_t n_ctx = q.shape()[0];
    const uint32_t n_embed = q.shape()[1];
    const uint32_t d_head = n_embed / n_head;

    const float *q_data = q.data_ptr<float>();
    const float *k_data = k.data_ptr<float>();

    qk_cache_.reshape({n_head, n_ctx, n_ctx});
    float *qk_data = qk_cache_.data_ptr<float>();

    qkv_cache_.reshape({n_ctx, n_embed});
    float *qkv_data = qkv_cache_.data_ptr<float>();
    const float *v_data = v.data_ptr<float>();

    float scale_factor = 1.0f / std::sqrt((float)d_head);
    if (qkv_cached_)
    {
        // SCALED DOT-PRODUCT: (Q @ K) * scale.
        for (uint32_t head = 0; head < n_head; head++)
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
        for (uint32_t head = 0; head < n_head; head++)
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
        for (uint32_t head = 0; head < n_head; head++)
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
        for (uint32_t head = 0; head < n_head; head++)
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
        uint32_t n_rows = n_head * n_ctx;
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
        for (uint32_t head = 0; head < n_head; head++)
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
                                               const uint32_t max_ctx,
                                               const uint32_t n_embed)
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

GPT2::GPT2(const std::string &fname)
{
    load_from_file(fname);
}

Tensor GPT2::logits(const Tensor &inp)
{
    Tensor logits;

    logits = res_.forward(wte_.forward(inp), wpe_.forward(inp.shape()[0]));
    for (auto &block : blocks_)
        logits = block.forward(logits);
    logits = ln_f_.forward(logits);
    logits = wte_.forward_project(logits);

    return logits;
}


void GPT2::show_performance() const
{
    std::cout << "\n";
    std::cout << "------------------------------------\n";
    std::cout << "LAYER/OP    TIME PER TOKEN  TIME TOTAL\n";
    std::cout << "------------------------------------\n";
    std::cout << "Embedding      | " << std::setw(2) << wte_.emb_time()/niter_      << "ms | " << wte_.emb_time()      << "ms\n";
    std::cout << "Embedding proj | " << std::setw(2) << wte_.emb_proj_time()/niter_ << "ms | " << wte_.emb_proj_time() << "ms\n";
    std::cout << "Pos embedding  | " << std::setw(2) << wpe_.time()/niter_          << "ms | " << wpe_.time()          << "ms\n";

    int64_t linear_time = 0;
    int64_t mlpp_time = 0;
    int64_t attn_lin = 0;
    int64_t attn_time = 0;
    int64_t ln_time = 0;
    int64_t gelu_time = 0;
    int64_t res_time = 0;
    for (const auto &block : blocks_)
    {
        linear_time += block.time_linear();
        attn_time += block.time_attn();
        ln_time += block.time_ln();
        gelu_time += block.time_gelu();
        mlpp_time += block.time_proj();
        attn_lin += block.time_attn_lin();
        res_time += block.time_res();
    }
    ln_time += ln_f_.time();
    res_time += res_.time();
    std::cout << "Linear (total) | " << std::setw(2) << linear_time/niter_ << "ms | " << linear_time << "ms\n";
    // std::cout << "Linear (qkv)   | " << std::setw(2) << attn_lin/niter_    << "ms | " << attn_lin    << "ms\n";
    // std::cout << "Linear (mlp)   | " << std::setw(2) << mlpp_time/niter_   << "ms | " << mlpp_time   << "ms\n";
    std::cout << "Attention      | " << std::setw(2) << attn_time/niter_   << "ms | " << attn_time   << "ms\n";
    std::cout << "Layer norm     | " << std::setw(2) << ln_time/niter_     << "ms | " << ln_time     << "ms\n";
    std::cout << "GELU           | " << std::setw(2) << gelu_time/niter_   << "ms | " << gelu_time   << "ms\n";
    std::cout << "Residual       | " << std::setw(2) << res_time/niter_    << "ms | " << res_time   << "ms\n";
    std::cout << "Sampler        | " << std::setw(2) << time_sample_ms_/niter_    << "ms | " << time_sample_ms_   << "ms\n";
    std::cout << "------------------------------------\n\n";
}

void GPT2::sample(const std::string &prompt, double temp, int max_iter)
{
    GTEN_CHECK(max_iter <= config_.n_ctx, "max_iter, %d, cannot exceed maximum context length, %ld.\n", max_iter, config_.n_ctx);

    time_sample_ms_ = 0;  // Reset for each call.
    niter_ = 0;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<int32_t> tokens = tokenizer.encode(prompt);
    tokens.reserve(max_iter);
    gten::Tensor logits;
    const int logits_size = 50257;
    std::vector<std::pair<double, int>> logits_probs;
    logits_probs.reserve(logits_size);

    const int eot_token = tokenizer.encode("<|endoftext|>")[0];
    const int initial_pos = tokens.size() - 1;
    const int n_iter = max_iter - tokens.size();
    // Use cerr because it is unbuffered.
    std::cerr << prompt;
    for (int i = initial_pos; i < n_iter; i++)
    {
        Tensor input(tokens.data(), {(uint32_t)tokens.size()}, Dtype::Int32);
        logits = this->logits(input);

        Timer timer(&time_sample_ms_);
        const float *logits_data = logits.data_ptr<float>();

        logits_probs.clear();
        for (int j = 0; j < logits_size; ++j)
            logits_probs.push_back(std::make_pair((double)logits_data[j] / temp, j));

        const int top_k = 16;
        
        // Select top k elements.
        std::partial_sort(
                logits_probs.begin(),
                logits_probs.begin() + top_k,
                logits_probs.end(),
                [](const std::pair<double, int> &rhs, const std::pair<double, int> &lhs) {
            return rhs.first > lhs.first;
        });
        logits_probs.resize(top_k);
        
        // compute softmax
        double sum_exp = 0;
        for (int j = 0; j < top_k; ++j)
        {
            logits_probs[j].first = std::exp(logits_probs[j].first);
            sum_exp += logits_probs[j].first;
        }
        for (int j = 0; j < top_k; ++j)
            logits_probs[j].first = logits_probs[j].first / sum_exp;

        std::vector<double> probs(logits_size, 0.0);
        for (int j = 0; j < top_k; j++)
        {
            const auto &prob_pair = logits_probs[j];
            probs[prob_pair.second] = prob_pair.first;
        }

        std::discrete_distribution dist(probs.begin(), probs.end());
        uint32_t maxi = dist(gen);
        if (maxi == eot_token)
            break;
        std::cerr << tokenizer.decode(maxi);
        tokens.push_back(maxi);

        niter_ += 1;
    }
    std::cerr << "\n";
}

/** File format.  TODO: Data ordering and matrix ordering.
 * The GPT2 gten file format is designed to be simple and minimal. We pack vocab,
 * config and layer weight data in a single binary file. The different sections have
 * names to allow for debugging.
 * 
 *  number |     section       | size in bytes
 *  ------------------------------------------
 *    1    |      magic        | 8
 *    2    |     n_vocab       | 8
 *    3    |      n_ctx        | 8
 *    4    |     n_embed       | 8
 *    5    |     n_layer       | 8
 *    6    |      n_head       | 8
 * 
 * for section in [vocab, wte, wpe, block_0, ... block_{n_layer-1}, ln_f]
 * 
 *    _    | section_name_size | 8
 *    _    | section name      | section_name_size
 *    _    | section_data_size | 8
 *    _    | section_data      | section_data_size
 *  -------------------------------------------
*/
void GPT2::load_from_file(const std::string &fname)
{
    std::ifstream fin(fname, std::ios::binary);
    GTEN_CHECK(fin.is_open(), "Failed to open model file: %s\n", fname.c_str());

    std::cout << "Loading from " << fname.c_str() << "\n";
    int64_t load_time;
    Timer timer(&load_time);

    int64_t magic;
    fin.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    GTEN_CHECK(magic == magic_number_, "Magic number in file %s does not match the expected one.\n", fname.c_str());

    config_ = GPT2Config();
    fin.read(reinterpret_cast<char*>(&config_), sizeof(config_));
    std::cout << config_;

    // Vocab
    std::string segment_name;
    int64_t segment_name_size;
    int64_t segment_size;
    fin.read(reinterpret_cast<char*>(&segment_name_size), sizeof(segment_name_size));
    segment_name.resize(segment_name_size);
    fin.read(reinterpret_cast<char*>(segment_name.data()), segment_name_size);
    fin.read(reinterpret_cast<char*>(&segment_size), sizeof(segment_size));
    // std::cout << "Reading segment: [" << segment_name << "](" << segment_size << " bytes)\n";

    tokenizer = std::move(GPT2Tokenizer(fin));

    // WTE
    fin.read(reinterpret_cast<char*>(&segment_name_size), sizeof(segment_name_size));
    segment_name.resize(segment_name_size);
    fin.read(reinterpret_cast<char*>(segment_name.data()), segment_name_size);
    fin.read(reinterpret_cast<char*>(&segment_size), sizeof(segment_size));
    Tensor wte_weight({(uint32_t)config_.n_vocab, (uint32_t)config_.n_embed}, Dtype::Float32);
    fin.read(reinterpret_cast<char*>(wte_weight.data_ptr<void>()), segment_size);
    wte_ = Embedding(wte_weight, config_.n_vocab, config_.n_ctx, config_.n_embed);
    // std::cout << "Reading segment: [" << segment_name << "](" << segment_size << " bytes)\n";

    // WPE
    fin.read(reinterpret_cast<char*>(&segment_name_size), sizeof(segment_name_size));
    segment_name.resize(segment_name_size);
    fin.read(reinterpret_cast<char*>(segment_name.data()), segment_name_size);
    fin.read(reinterpret_cast<char*>(&segment_size), sizeof(segment_size));
    Tensor wpe_weight({(uint32_t)config_.n_ctx, (uint32_t)config_.n_embed}, Dtype::Float32);
    fin.read(reinterpret_cast<char*>(wpe_weight.data_ptr<void>()), segment_size);
    wpe_ = PosEmbedding(wpe_weight, config_.n_ctx, config_.n_embed);
    // std::cout << "Reading segment: [" << segment_name << "](" << segment_size << " bytes)\n";

    // BLOCKS
    blocks_ = std::vector<ResidualAttentionBlock>();
    blocks_.reserve(config_.n_layer);
    for (int64_t i = 0; i < config_.n_layer; i++)
    {
        fin.read(reinterpret_cast<char*>(&segment_name_size), sizeof(segment_name_size));
        segment_name.resize(segment_name_size);
        fin.read(reinterpret_cast<char*>(segment_name.data()), segment_name_size);
        fin.read(reinterpret_cast<char*>(&segment_size), sizeof(segment_size));
        // std::cout << "Reading segment: [" << segment_name << "](" << segment_size << " bytes)\n";

        Tensor qw({(uint32_t)config_.n_embed, (uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor qb({(uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor kw({(uint32_t)config_.n_embed, (uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor kb({(uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor vw({(uint32_t)config_.n_embed, (uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor vb({(uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor attn_projw({(uint32_t)config_.n_embed, (uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor attn_projb({(uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor ln_1w({(uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor ln_1b({(uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor mlp_fcw({4 * (uint32_t)config_.n_embed, (uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor mlp_fcb({4 * (uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor mlp_projw({(uint32_t)config_.n_embed, 4 * (uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor mlp_projb({(uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor ln_2w({(uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor ln_2b({(uint32_t)config_.n_embed}, Dtype::Float32);

        fin.read(reinterpret_cast<char*>(qw.data_ptr<void>()),         qw.numel()         * qw.bytes_per_item());
        fin.read(reinterpret_cast<char*>(qb.data_ptr<void>()),         qb.numel()         * qb.bytes_per_item());
        fin.read(reinterpret_cast<char*>(kw.data_ptr<void>()),         kw.numel()         * kw.bytes_per_item());
        fin.read(reinterpret_cast<char*>(kb.data_ptr<void>()),         kb.numel()         * kb.bytes_per_item());
        fin.read(reinterpret_cast<char*>(vw.data_ptr<void>()),         vw.numel()         * vw.bytes_per_item());
        fin.read(reinterpret_cast<char*>(vb.data_ptr<void>()),         vb.numel()         * vb.bytes_per_item());
        fin.read(reinterpret_cast<char*>(attn_projw.data_ptr<void>()), attn_projw.numel() * attn_projw.bytes_per_item());
        fin.read(reinterpret_cast<char*>(attn_projb.data_ptr<void>()), attn_projb.numel() * attn_projb.bytes_per_item());
        fin.read(reinterpret_cast<char*>(ln_1w.data_ptr<void>()),      ln_1w.numel()      * ln_1w.bytes_per_item());
        fin.read(reinterpret_cast<char*>(ln_1b.data_ptr<void>()),      ln_1b.numel()      * ln_1b.bytes_per_item());
        fin.read(reinterpret_cast<char*>(mlp_fcw.data_ptr<void>()),    mlp_fcw.numel()    * mlp_fcw.bytes_per_item());
        fin.read(reinterpret_cast<char*>(mlp_fcb.data_ptr<void>()),    mlp_fcb.numel()    * mlp_fcb.bytes_per_item());
        fin.read(reinterpret_cast<char*>(mlp_projw.data_ptr<void>()),  mlp_projw.numel()  * mlp_projw.bytes_per_item());
        fin.read(reinterpret_cast<char*>(mlp_projb.data_ptr<void>()),  mlp_projb.numel()  * mlp_projb.bytes_per_item());
        fin.read(reinterpret_cast<char*>(ln_2w.data_ptr<void>()),      ln_2w.numel()      * ln_2w.bytes_per_item());
        fin.read(reinterpret_cast<char*>(ln_2b.data_ptr<void>()),      ln_2b.numel()      * ln_2b.bytes_per_item());

        const Linear query(qw, qb, config_.n_ctx);
        const Linear key(kw, kb, config_.n_ctx);
        const Linear value(vw, vb, config_.n_ctx);
        const Linear out_proj(attn_projw, attn_projb, config_.n_ctx);
        const MultiHeadSelfAttn self_attn(query, key, value, out_proj, config_.n_ctx, config_.n_embed);
        const LayerNorm ln_1(ln_1w, ln_1b, config_.n_ctx, config_.n_embed);
        const Linear mlp_fc(mlp_fcw, mlp_fcb, config_.n_ctx);
        const Linear mlp_proj(mlp_projw, mlp_projb, config_.n_ctx);
        const LayerNorm ln_2(ln_2w, ln_2b, config_.n_ctx, config_.n_embed);
        const GELU gelu(config_.n_ctx, config_.n_embed * 4);

        blocks_.push_back(ResidualAttentionBlock(self_attn, ln_1, mlp_fc, mlp_proj, ln_2, gelu, config_.n_ctx, config_.n_embed));
    }
    
    // LN_F
    fin.read(reinterpret_cast<char*>(&segment_name_size), sizeof(segment_name_size));
    segment_name.resize(segment_name_size);
    fin.read(reinterpret_cast<char*>(segment_name.data()), segment_name_size);
    fin.read(reinterpret_cast<char*>(&segment_size), sizeof(segment_size));
    // std::cout << "Reading segment: [" << segment_name << "](" << segment_size << " bytes)\n";

    Tensor ln_fw({(uint32_t)config_.n_embed}, Dtype::Float32);
    Tensor ln_fb({(uint32_t)config_.n_embed}, Dtype::Float32);
    fin.read(reinterpret_cast<char*>(ln_fw.data_ptr<void>()), ln_fw.numel() * ln_fw.bytes_per_item());
    fin.read(reinterpret_cast<char*>(ln_fb.data_ptr<void>()), ln_fb.numel() * ln_fb.bytes_per_item());

    ln_f_ = LayerNorm(ln_fw, ln_fb, config_.n_ctx, config_.n_embed);

    res_ = Residual(config_.n_ctx, config_.n_embed);

    timer.stop();
    std::cout << "Load time: " << load_time << " ms\n\n";
}

} // namespace gten
