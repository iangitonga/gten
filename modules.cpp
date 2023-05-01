#include "modules.h"

#include <cmath>
#include <iostream>

namespace gten
{

Embedding::Embedding(InferenceMode mode, const Tensor& weight, int32_t max_ctx)
    : inference_mode_(mode), weight_(weight), n_vocab_(weight.size(0)), max_ctx_(max_ctx),
      n_embed_(weight.size(1)), emb_out_(Tensor({max_ctx_, n_embed_}, mode)),
      proj_out_(Tensor({1, n_vocab_}, kFloat32))
{
}

Tensor Embedding::forward(const Tensor& inp)
{
    Timer timer(&time_embed_ms_);

    if (inference_mode_ == kFloat16)
        return forward_impl<Float16>(inp);
    else
        return forward_impl<Float32>(inp);
}

template<typename scalar_t>
Tensor Embedding::forward_impl(const Tensor& inp)
{
    emb_out_.resize({inp.numel(), n_embed_});
    scalar_t* out_data = emb_out_.data_ptr<scalar_t>();
    const Int32* inp_data = inp.data_ptr<Int32>();
    scalar_t* weight_data = weight_.data_ptr<scalar_t>();

    if (emb_out_cached_) {
        const int token_i = inp.numel() - 1;
        const int emb_i = inp_data[token_i] * n_embed_;
        void *src = reinterpret_cast<void*>(weight_data + emb_i);
        void *dest = reinterpret_cast<void*>(out_data + token_i * n_embed_);
        std::memcpy(dest, src, n_embed_ * weight_.itemsize());
    }
    else {
        emb_out_cached_ = true;

        const int ntokens = inp.numel();
        for (int token_i = 0; token_i < ntokens; token_i++)
        {
            int emb_i = inp_data[token_i] * n_embed_;
            void *src = reinterpret_cast<void*>(weight_data + emb_i);
            void *dest = reinterpret_cast<void*>(out_data + token_i * n_embed_);
            std::memcpy(dest, src, n_embed_ * weight_.itemsize());
        }
    }

    return emb_out_;    
}


Tensor Embedding::forward_proj(const Tensor &inp)
{
    Timer timer(&time_project_ms_);

    if (inference_mode_ == kFloat16)
        return forward_proj_impl<Float16>(inp);
    else
        return forward_proj_impl<Float32>(inp);
}

template<typename scalar_t>
Tensor Embedding::forward_proj_impl(const Tensor& inp) {
    // Output probs must be float32.
    Float32* out_data = proj_out_.data_ptr<Float32>();
    const scalar_t* inp_data = inp.data_ptr<scalar_t>();
    const scalar_t* weight_data = weight_.data_ptr<scalar_t>();

    const int n_ctx = inp.size(0);
    const int n_embed = inp.size(1);
    const int ctx_i = n_ctx - 1;

#if GTEN_SIMD
    for (int emb_i = 0; emb_i < n_vocab_; emb_i++)
    {
        Vec8_f32 dot_accum = { vec8_f32_setzero() };
        for (int el = 0; el < n_embed; el += 8) {
            Vec8_f32 iv = vec8_f32_load(inp_data + (ctx_i * n_embed + el));
            Vec8_f32 wv = vec8_f32_load(weight_data + (emb_i * n_embed + el));
            dot_accum = vec8_f32_fma(iv, wv, dot_accum);
        }
        out_data[emb_i] = vec8_f32_sum(dot_accum);
    }
#else
    for (int emb_i = 0; emb_i < n_vocab_; emb_i++)
    {
        Float32 dot_prod = 0.0f;
        for (int el = 0; el < n_embed; el++) {
            Float32 inp_w = fpcvt_to_fp32<scalar_t>(inp_data[ctx_i * n_embed + el]);
            Float32 w = fpcvt_to_fp32<scalar_t>(weight_data[emb_i * n_embed + el]);
            dot_prod += inp_w * w;
        }
        out_data[emb_i] = dot_prod;
    }
#endif

    return proj_out_;  
}

PosEmbedding::PosEmbedding(InferenceMode mode, const Tensor &weight, int32_t max_ctx)
    : inference_mode_(mode), weight_(weight), max_ctx_(max_ctx), n_embed_( weight.size(1)),
      out_(Tensor({max_ctx, n_embed_}, mode))
{
}

Tensor PosEmbedding::forward(int32_t n_ctx)
{
    Timer timer(&time_ms_);
    
    if (inference_mode_ == kFloat16)
        return forward_impl<Float16>(n_ctx);
    else
        return forward_impl<Float32>(n_ctx);
}

template<typename scalar_t>
Tensor PosEmbedding::forward_impl(int32_t n_ctx)
{
    // TODO: Could we return a tensor sharing data with table data.
    out_.resize({n_ctx, n_embed_});
    scalar_t* out_data = out_.data_ptr<scalar_t>();
    const scalar_t* weight_data = weight_.data_ptr<scalar_t>();

    if (out_cached_) {
        const int data_offset = (n_ctx - 1) * n_embed_;
        size_t bytes_to_copy = n_embed_ * weight_.itemsize();
        const void* src_ptr = reinterpret_cast<const void*>(weight_data + data_offset);
        void* dest_ptr = reinterpret_cast<void*>(out_data + data_offset);
        std::memcpy(dest_ptr, src_ptr, bytes_to_copy);
    }
    else {
        out_cached_ = true;

        size_t bytes_to_copy = n_ctx * n_embed_ * weight_.itemsize();
        const void* src_ptr = reinterpret_cast<const void*>(weight_data);
        void* dest_ptr = reinterpret_cast<void*>(out_data);
        std::memcpy(dest_ptr, src_ptr, bytes_to_copy);
    }
    return out_;
}

LayerNorm::LayerNorm(InferenceMode mode, const Tensor &weight, const Tensor &bias, int32_t max_ctx)
    : inference_mode_(mode), weight_(weight), bias_(bias), max_ctx_(max_ctx),
      n_embed_(weight.size(0)), out_(Tensor({max_ctx, n_embed_}, mode))
{
}

Tensor LayerNorm::forward(const Tensor &inp)
{
    Timer timer(&time_ms_);

    if (inference_mode_ == kFloat16)
        return forward_impl<Float16>(inp);
    else
        return forward_impl<Float32>(inp);
}

template<typename scalar_t>
Tensor LayerNorm::forward_impl(const Tensor &inp)
{
    out_.resize(inp.shape());
    scalar_t* out_data = out_.data_ptr<scalar_t>();
    const scalar_t* inp_data = inp.data_ptr<scalar_t>();
    const scalar_t* weight_data = weight_.data_ptr<scalar_t>();
    const scalar_t* bias_data = bias_.data_ptr<scalar_t>();

    const int n_ctx = inp.size(0);

    if (out_cached_)
    {
        const int ctx_offset = (n_ctx - 1) * n_embed_;
        // Mean calculation.
        Float32 mean_accum = 0.0f;
        for (int el = 0; el < n_embed_; el++)
            mean_accum += fpcvt_to_fp32<scalar_t>(inp_data[el + ctx_offset]);
        Float32 mean = mean_accum / (Float32)n_embed_;

        // Standard deviation calculation.
        Float32 variance_accum = 0.0f;
        for (int el = 0; el < n_embed_; el++)
            variance_accum += std::pow(fpcvt_to_fp32<scalar_t>(inp_data[el + ctx_offset]) - mean, 2);
        Float32 std_dev = std::sqrt(variance_accum / (Float32)n_embed_);

        // Normalization.
        for (int el = 0; el < n_embed_; el++) {
            Float32 unnormalized = fpcvt_to_fp32<scalar_t>(inp_data[el + ctx_offset]);
            Float32 w = fpcvt_to_fp32<scalar_t>(weight_data[el]);
            Float32 b = fpcvt_to_fp32<scalar_t>(bias_data[el]);
            Float32 normalized = ((unnormalized - mean) / (std_dev + eps_)) * w + b;
            out_data[el + ctx_offset] = fpcvt_from_fp32<scalar_t>(normalized);
        }
    }
    else
    {
        out_cached_ = true;

        for (int ctx_i = 0; ctx_i < n_ctx; ctx_i++)
        {
            const int ctx_offset = ctx_i * n_embed_;

            // Mean calculation.
            Float32 mean_accum = 0.0f;
            for (int el = 0; el < n_embed_; el++)
                mean_accum += fpcvt_to_fp32<scalar_t>(inp_data[el + ctx_offset]);
            Float32 mean = mean_accum / (Float32)n_embed_;

            // Standard deviation calculation.
            Float32 variance_accum = 0.0f;
            for (int el = 0; el < n_embed_; el++)
                variance_accum += std::pow(fpcvt_to_fp32<scalar_t>(inp_data[el + ctx_offset]) - mean, 2);
            Float32 std_dev = std::sqrt(variance_accum / (Float32)n_embed_);

            // Normalization.
            for (int el = 0; el < n_embed_; el++) {
                Float32 unnormalized = fpcvt_to_fp32<scalar_t>(inp_data[el + ctx_offset]);
                Float32 w = fpcvt_to_fp32<scalar_t>(weight_data[el]);
                Float32 b = fpcvt_to_fp32<scalar_t>(bias_data[el]);
                // Epsilon added to standard deviation prevents div by zero.
                Float32 normalized = ((unnormalized - mean) / (std_dev + eps_)) * w + b;
                out_data[el + ctx_offset] = fpcvt_from_fp32<scalar_t>(normalized);
            }
        }
    }
    return out_;
}

GELU::GELU(InferenceMode mode, int32_t max_ctx, int32_t n_out)
    : inference_mode_(mode), n_out_(n_out), out_(Tensor({max_ctx, n_out_}, mode))
{
}

Tensor GELU::forward(const Tensor &inp)
{
    Timer timer(&time_ms_);

    if (inference_mode_ == kFloat16)
        return forward_impl<Float16>(inp);
    else
        return forward_impl<Float32>(inp);
}

template <typename scalar_t>
Tensor GELU::forward_impl(const Tensor& inp)
{
    const int n_ctx = inp.size(0);

    out_.resize({n_ctx, n_out_});
    scalar_t* out_data = out_.data_ptr<scalar_t>();
    const scalar_t* inp_data = inp.data_ptr<scalar_t>();

    const int ne = inp.numel();
    if (out_cached_) {
        const int start_i = (n_ctx - 1) * n_out_;
        Float32 x;
        for (int i = start_i; i < ne; ++i) {
            x = fpcvt_to_fp32<scalar_t>(inp_data[i]);
            Float32 res = 0.5 * x * (1.0f + std::tanh(std::sqrt(2.0f / 3.141592653589793f)
                                  * (x + 0.044715f * std::pow(x, 3.0f))));
            out_data[i] = fpcvt_from_fp32<scalar_t>(res);
        }
    }
    else {
        out_cached_ = true;
        Float32 x;
        for (int i = 0; i < ne; ++i) {
            x = fpcvt_to_fp32<scalar_t>(inp_data[i]);
            Float32 res = 0.5 * x * (1.0f + std::tanh(std::sqrt(2.0f / 3.141592653589793f)
                                  * (x + 0.044715f * std::pow(x, 3.0f))));
            out_data[i] = fpcvt_from_fp32<scalar_t>(res);
        }
    }
    return out_;
}

Residual::Residual(InferenceMode mode, int32_t max_ctx, int32_t n_embed)
    : inference_mode_(mode), out_(Tensor({max_ctx, n_embed}, mode))
{
}

Tensor Residual::forward(const Tensor& inp0, const Tensor& inp1)
{
    GTEN_ASSERT(inp0.numel() == inp1.numel(), "Addition of diff length tensors not allowed.");

    Timer timer(&time_ms_);

    if (inference_mode_ == kFloat16)
        return forward_impl<Float16>(inp0, inp1);
    else
        return forward_impl<Float32>(inp0, inp1);
}

template<typename scalar_t>
Tensor Residual::forward_impl(const Tensor& inp0, const Tensor& inp1)
{
    const int n_ctx = inp0.size(0);
    const int n_embed = inp0.size(1);

    out_.resize({n_ctx, n_embed});
    scalar_t* out_data = out_.data_ptr<scalar_t>();
    const scalar_t* inp0_data = inp0.data_ptr<scalar_t>();
    const scalar_t* inp1_data = inp1.data_ptr<scalar_t>();

#if GTEN_SIMD
    if (out_cached_)
    {
        uint32_t n_iter = n_embed;
        uint32_t offset = inp0.numel() - n_embed;
        for (uint32_t i = 0; i < n_iter; i += 8) {
            Vec8_f32 v0 = vec8_f32_load(inp0_data + offset + i);
            Vec8_f32 v1 = vec8_f32_load(inp1_data + offset + i);
            Vec8_f32 res = vec8_f32_add(v0, v1);
            vec8_f32_store(res, out_data + offset + i);
        }
    }
    else {
        out_cached_ = true;
        int n_iter = inp0.numel();
        for (int i = 0; i < n_iter; i += 8) {
            Vec8_f32 v0 = vec8_f32_load(inp0_data + i);
            Vec8_f32 v1 = vec8_f32_load(inp1_data + i);
            Vec8_f32 res = vec8_f32_add(v0, v1);
            vec8_f32_store(res, out_data + i);
        }
    }
#else
    if (out_cached_)
    {
        int n_iter = n_embed;
        int offset = inp0.numel() - n_embed;
        for (int i = 0; i < n_iter; i++) {
            Float32 inp0_w = fpcvt_to_fp32<scalar_t>(inp0_data[i + offset]);
            Float32 inp1_w = fpcvt_to_fp32<scalar_t>(inp1_data[i + offset]);
            out_data[i + offset] = fpcvt_from_fp32<scalar_t>(inp0_w + inp1_w);
        }
    }   
    else {
        out_cached_ = true;
        int n_iter = inp0.numel();
        for (int i = 0; i < n_iter; i++) {
            Float32 inp0_w = fpcvt_to_fp32<scalar_t>(inp0_data[i]);
            Float32 inp1_w = fpcvt_to_fp32<scalar_t>(inp1_data[i]);
            out_data[i] = fpcvt_from_fp32<scalar_t>(inp0_w + inp1_w);
        }
    }
#endif
    
    return out_;
}

Linear::Linear(InferenceMode mode, const Tensor &weight, const Tensor &bias, int32_t max_ctx)
    : inference_mode_(mode), weight_(weight), bias_(bias),
      out_(Tensor({max_ctx, weight_.size(0)}, mode))
{
}

Tensor Linear::forward(const Tensor &inp)
{
    Timer timer(&time_ms_);

    if (inference_mode_ == kFloat16)
        return forward_impl<Float16>(inp);
    else
        return forward_impl<Float32>(inp);
}

template<typename scalar_t>
Tensor Linear::forward_impl(const Tensor& inp)
{
    const int n_ctx = inp.size(0);
    const int n_embed = inp.size(1);
    const int n_out = weight_.size(0);
    
    out_.resize({n_ctx, n_out});
    scalar_t* out_data = out_.data_ptr<scalar_t>();
    const scalar_t* weight_data = weight_.data_ptr<scalar_t>();
    const scalar_t* bias_data = bias_.data_ptr<scalar_t>();
    const scalar_t* inp_data = inp.data_ptr<scalar_t>();

#if GTEN_SIMD

    if (out_cached_)
    {
        const int ctx_i = n_ctx - 1;
        for (int out_i = 0; out_i < n_out; out_i++) {
            Vec8_f32 dot_prod_accum = vec8_f32_setzero();
            for (int el = 0; el < n_embed; el += 8) {
                Vec8_f32 iv = vec8_f32_load(inp_data + (ctx_i * n_embed + el));
                Vec8_f32 wv = vec8_f32_load(weight_data + (out_i * n_embed + el));
                dot_prod_accum = vec8_f32_fma(iv, wv, dot_prod_accum);
            }
            Float32 bias = fpcvt_to_fp32<scalar_t>(bias_data[out_i]);
            Float32 res =  vec8_f32_sum(dot_prod_accum) + bias;
            out_data[ctx_i * n_out + out_i] = fpcvt_from_fp32<scalar_t>(res);
        }
    }
    else
    {
        out_cached_ = true;
        for (int ctx_i = 0; ctx_i < n_ctx; ctx_i++) {
            for (int out_i = 0; out_i < n_out; out_i++) {
                Vec8_f32 dot_prod_accum = vec8_f32_setzero();
                for (int el = 0; el < n_embed; el += 8) {
                    Vec8_f32 iv = vec8_f32_load(inp_data + (ctx_i * n_embed + el));
                    Vec8_f32 wv = vec8_f32_load(weight_data + (out_i * n_embed + el));
                    dot_prod_accum = vec8_f32_fma(iv, wv, dot_prod_accum);
                }
                Float32 bias = fpcvt_to_fp32<scalar_t>(bias_data[out_i]);
                Float32 res =  vec8_f32_sum(dot_prod_accum) + bias;
                out_data[ctx_i * n_out + out_i] = fpcvt_from_fp32<scalar_t>(res);
            }
        }
    }

#else

    if (out_cached_)
    {
        const int ctx_i = n_ctx - 1;
        for (int out_i = 0; out_i < n_out; out_i++) {
            Float32 dot_prod = 0.0f;
            for (int el = 0; el < n_embed; el++) {
                Float32 inp_w = fpcvt_to_fp32<scalar_t>(inp_data[ctx_i * n_embed + el]);
                Float32 w = fpcvt_to_fp32<scalar_t>(weight_data[out_i * n_embed + el]);
                dot_prod += inp_w * w;
            }
            Float32 bias = fpcvt_to_fp32<scalar_t>(bias_data[out_i]);
            out_data[ctx_i * n_out + out_i] = fpcvt_from_fp32<scalar_t>(dot_prod + bias);
        }
    }
    else
    {
        out_cached_ = true;
        for (int ctx_i = 0; ctx_i < n_ctx; ctx_i++) {
            for (int out_i = 0; out_i < n_out; out_i++) {
                Float32 dot_prod = 0.0f;
                for (int el = 0; el < n_embed; el++) {
                    Float32 inp_w = fpcvt_to_fp32<scalar_t>(inp_data[ctx_i * n_embed + el]);
                    Float32 w = fpcvt_to_fp32<scalar_t>(weight_data[out_i * n_embed + el]);
                    dot_prod += inp_w * w;
                }
                Float32 bias = fpcvt_to_fp32<scalar_t>(bias_data[out_i]);
                out_data[ctx_i * n_out + out_i] = fpcvt_from_fp32<scalar_t>(dot_prod + bias);
            }
        }
    }

#endif

    return out_;
}

MultiHeadSelfAttn::MultiHeadSelfAttn(InferenceMode mode,
                                     const Linear& query,
                                     const Linear& key,
                                     const Linear& value,
                                     const Linear& out_proj,
                                     int32_t max_ctx,
    int32_t n_embed, int32_t n_head)
    : inference_mode_(mode), query_(query), key_(key), value_(value), qkv_proj_(out_proj),
      n_head_(n_head), qk_cache_(Tensor({n_head, max_ctx, max_ctx}, mode)),
      qkv_cache_(Tensor({max_ctx, n_embed}, mode))
{
    if (mode == kFloat16) {
        Float16 *qk_cache_data = qk_cache_.data_ptr<Float16>();
        const int ne = qk_cache_.numel();
        const Float16 inf = fp32_to_fp16(-INFINITY);
        for (int i = 0; i < ne; i++)
            qk_cache_data[i] = inf;
    }
    else {
        Float32 *qk_cache_data = qk_cache_.data_ptr<Float32>();
        const int ne = qk_cache_.numel();
        for (int i = 0; i < ne; i++)
            qk_cache_data[i] = -INFINITY;
    }
}

Tensor MultiHeadSelfAttn::forward(const Tensor &inp)
{
    const int n_ctx = inp.size(0);
    const int n_embed = inp.size(1);

    Tensor q = query_.forward(inp);
    Tensor k = key_.forward(inp);
    Tensor v = value_.forward(inp);

    const Tensor qkv = (inference_mode_ == kFloat16)
                       ? qkv_attn<Float16>(q, k, v) : qkv_attn<Float32>(q, k, v);
    const Tensor out = qkv_proj_.forward(qkv);
    return out;
}

template<typename scalar_t>
Tensor MultiHeadSelfAttn::qkv_attn(const Tensor &q, const Tensor &k, const Tensor &v)
{
    Timer timer(&time_attn_ms_);

    const int n_ctx = q.size(0);
    const int n_embed = q.size(1);
    const int d_head = n_embed / n_head_;

    const scalar_t* q_data = q.data_ptr<scalar_t>();
    const scalar_t* k_data = k.data_ptr<scalar_t>();

    qk_cache_.resize({n_head_, n_ctx, n_ctx});
    scalar_t *qk_data = qk_cache_.data_ptr<scalar_t>();

    qkv_cache_.resize({n_ctx, n_embed});
    scalar_t *qkv_data = qkv_cache_.data_ptr<scalar_t>();
    const scalar_t *v_data = v.data_ptr<scalar_t>();

    Float32 scale_factor = 1.0f / std::sqrt((Float32)d_head);
    if (qkv_cached_)
    {
        // SCALED DOT-PRODUCT: (Q @ K) * scale.
        for (int head = 0; head < n_head_; head++)
        {
            const int q_row = n_ctx - 1;
            for (int k_col = 0; k_col < n_ctx; k_col++) {
                Float32 dot_prod = 0.0f;
                for (int el = 0; el < d_head; el++) {
                    int q_idx = head * d_head + q_row * n_embed + el;
                    int k_idx = head * d_head + k_col * n_embed + el;
                    Float32 q_w = fpcvt_to_fp32<scalar_t>(q_data[q_idx]);
                    Float32 k_w = fpcvt_to_fp32<scalar_t>(k_data[k_idx]);
                    dot_prod += q_w * k_w * scale_factor;
                }
                qk_data[head * n_ctx * n_ctx + q_row * n_ctx + k_col] = fpcvt_from_fp32<scalar_t>(dot_prod);
            }
        }

        // SOFTMAX
        for (int head = 0; head < n_head_; head++)
        {
            const int row = n_ctx - 1;
            const int base_i = head * n_ctx * n_ctx + row * n_ctx;
            Float32 max = -INFINITY;
            for (int el = 0; el < n_ctx; el++) {
                Float32 val = fpcvt_to_fp32<scalar_t>(qk_data[base_i + el]);
                if (val > max)
                    max = val;
            }

            Float32 sum_exp = 0;
            for (int el = 0; el < n_ctx; el++) {
                int idx = base_i + el;
                Float32 val = fpcvt_to_fp32<scalar_t>(qk_data[idx]);
                qk_data[idx] = fpcvt_from_fp32<scalar_t>(std::exp(val - max));
                sum_exp += fpcvt_to_fp32<scalar_t>(qk_data[idx]);
            }

            for (int el = 0; el < n_ctx; el++) {
                Float32 qk_w = fpcvt_to_fp32<scalar_t>(qk_data[base_i + el]);
                qk_data[base_i + el] = fpcvt_from_fp32<scalar_t>(qk_w / sum_exp);
            }
        }

        // ATTENTION: QK @ V
        for (int head = 0; head < n_head_; head++)
        {
            const int qk_row = n_ctx - 1;
            for (int v_col = 0; v_col < d_head; v_col++) {
                Float32 dot_prod = 0;
                for (int el = 0; el < n_ctx; el++) {
                    int qk_idx = head * n_ctx * n_ctx + qk_row * n_ctx + el;
                    int v_idx = head * d_head + el * n_embed + v_col;
                    Float32 qk_w = fpcvt_to_fp32<scalar_t>(qk_data[qk_idx]);
                    Float32 v_w = fpcvt_to_fp32<scalar_t>(v_data[v_idx]);
                    dot_prod += qk_w * v_w;
                }
                qkv_data[head * d_head + qk_row * n_embed + v_col] = fpcvt_from_fp32<scalar_t>(dot_prod);
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
                // `non_masked_prods` represents the number of dot products that will not be masked
                // and therefore must be computed. This allows us to skip unecessary dot products.
                int non_masked_prods = q_row + 1;
                for (int k_col = 0; k_col < non_masked_prods; k_col++) {
                    Float32 dot_prod = 0.0f;
                    for (int el = 0; el < d_head; el++) {
                        int q_idx = head * d_head + q_row * n_embed + el;
                        int k_idx = head * d_head + k_col * n_embed + el;
                        Float32 q_w = fpcvt_to_fp32<scalar_t>(q_data[q_idx]);
                        Float32 k_w = fpcvt_to_fp32<scalar_t>(k_data[k_idx]);
                        dot_prod += q_w * k_w * scale_factor;
                    }
                    qk_data[head * n_ctx * n_ctx + q_row * n_ctx + k_col] = fpcvt_from_fp32<scalar_t>(dot_prod);
                }

            }
        }

        // SOFTMAX
        int n_rows = n_head_ * n_ctx;
        for (int row = 0; row < n_rows; row++)
        {
            Float32 max = -INFINITY;
            for (int el = 0; el < n_ctx; el++) {
                Float32 val = fpcvt_to_fp32<scalar_t>(qk_data[row * n_ctx + el]);
                if (val > max)
                    max = val;
            }

            Float32 sum_exp = 0;
            for (int el = 0; el < n_ctx; el++) {
                int idx = row * n_ctx + el;
                Float32 val = fpcvt_to_fp32<scalar_t>(qk_data[idx]);
                qk_data[idx] = fpcvt_from_fp32<scalar_t>(std::exp(val - max));
                sum_exp += fpcvt_to_fp32<scalar_t>(qk_data[idx]);
            }

            for (int el = 0; el < n_ctx; el++) {
                Float32 qk_w =  fpcvt_to_fp32<scalar_t>(qk_data[row * n_ctx + el]);
                qk_data[row * n_ctx + el] = fpcvt_from_fp32<scalar_t>(qk_w / sum_exp);
            }
        }

        // ATTENTION: QK @ V
        for (int head = 0; head < n_head_; head++)
        {
            for (int qk_row = 0; qk_row < n_ctx; qk_row++){
                for (int v_col = 0; v_col < d_head; v_col++) {
                    Float32 dot_prod = 0;
                    for (int el = 0; el < n_ctx; el++) {
                        int qk_idx = head * n_ctx * n_ctx + qk_row * n_ctx + el;
                        int v_idx = head * d_head + el * n_embed + v_col;
                        Float32 qk_w =fpcvt_to_fp32<scalar_t>(qk_data[qk_idx]);
                        Float32 v_w = fpcvt_to_fp32<scalar_t>(v_data[v_idx]);
                        dot_prod += qk_w * v_w;
                    }
                    qkv_data[head * d_head + qk_row * n_embed + v_col] = fpcvt_from_fp32<scalar_t>(dot_prod);
                }
            }
        }
    }

    return qkv_cache_;
}

ResidualAttentionBlock::ResidualAttentionBlock(InferenceMode mode,
                                               const MultiHeadSelfAttn &attn,
                                               const LayerNorm &ln_1,
                                               const Linear &mlp_fc,
                                               const Linear &mlp_proj,
                                               const LayerNorm &ln_2,
                                               const GELU &gelu,
                                               int32_t max_ctx,
                                               int32_t n_embed)
    : attn_(attn), ln_1_(ln_1), mlp_fc_(mlp_fc), mlp_proj_(mlp_proj), ln_2_(ln_2),
      gelu_(gelu), inp_res_(Residual(mode, max_ctx, n_embed)),
      attn_res_(Residual(mode, max_ctx, n_embed))
{
}

Tensor ResidualAttentionBlock::forward(const Tensor &inp)
{
    Tensor attn = inp_res_.forward(inp, attn_.forward(ln_1_.forward(inp)));
    Tensor out = attn_res_.forward(attn, mlp_proj_.forward(gelu_.forward(mlp_fc_.forward(ln_2_.forward(attn)))));
    return out;
}

} // namespace gten
