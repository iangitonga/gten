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

#define GTEN_CHECK_INP_CTX_SIZE(inp_ctx_size, max_ctx_size)                  \
    GTEN_ASSERT(                                                             \
        inp_ctx_size <= max_ctx_size,                                        \
        "The given input's context size=%d exceeds max context size of %d.", \
        inp_ctx_size,                                                        \
        max_ctx_size)



namespace gten
{

Embedding::Embedding(const Tensor& weight, AcvConfig acv, int32_t max_ctx)
    : weight_{weight}, n_vocab_{weight.size(0)},
      max_ctx_{max_ctx}, n_embed_{weight.size(1)},
      emb_out_{Tensor{{max_ctx_, n_embed_}, acv.dtype, acv.scale, acv.zerop}},
      proj_out_{Tensor{{1, n_vocab_}, kFloat32}}
{
    GTEN_CHECK_NDIMS_EQUAL(weight_.ndims(), 2);
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
        for (int token_i = 0; token_i < ntokens; token_i++) {
            int emb_i = inp_data[token_i] * n_embed_;
            void *src = reinterpret_cast<void*>(weight_data + emb_i);
            void *dest = reinterpret_cast<void*>(out_data + token_i * n_embed_);
            std::memcpy(dest, src, n_embed_ * weight_.itemsize());
        }
    }

    return emb_out_;    
}

template<>
Tensor Embedding::forward_impl<Qint8>(const Tensor& inp)
{
    emb_out_.resize({inp.numel(), n_embed_});
    Float32* out_data = emb_out_.data_ptr<Float32>();
    const Int32* inp_data = inp.data_ptr<Int32>();
    Qint8* weight_data = weight_.data_ptr<Qint8>();

    if (emb_out_cached_) {
        const int token_i = inp.numel() - 1;
        const int emb_i = inp_data[token_i] * n_embed_;
        for (int i = 0; i < n_embed_; i++)
            out_data[token_i * n_embed_ + i] = weight_.deq(weight_data[emb_i + i]);
    }
    else {
        emb_out_cached_ = true;

        const int ntokens = inp.numel();
        for (int token_i = 0; token_i < ntokens; token_i++)
        {
            int emb_i = inp_data[token_i] * n_embed_;
            for (int i = 0; i < n_embed_; i++)
                out_data[token_i * n_embed_ + i] = weight_.deq(weight_data[emb_i + i]);
        }
    }

    return emb_out_;    
}

Tensor Embedding::forward(const Tensor& inp)
{
    Timer timer{&time_embed_ms_};

    GTEN_CHECK_DTYPE_EQUAL(inp.dtype(), kInt32);
    GTEN_CHECK_NDIMS_EQUAL(inp.ndims(), 1);
    GTEN_CHECK_INP_CTX_SIZE(inp.numel(), max_ctx_);

    if (weight_.dtype() == kQint8)
        return forward_impl<Qint8>(inp);
    else if (weight_.dtype() == kFloat16)
        return forward_impl<Float16>(inp);
    else
        return forward_impl<Float32>(inp);
}

template<typename scalar_t>
Tensor Embedding::forward_proj_impl(const Tensor& inp) {
    // Output probs must be float32.
    Float32* out_data = proj_out_.data_ptr<Float32>();
    const scalar_t* inp_data = inp.data_ptr<scalar_t>();
    const scalar_t* weight_data = weight_.data_ptr<scalar_t>();

    const int ctx_i = inp.size(0) - 1;
    const int n_embed = inp.size(1);

    for (int emb_i = 0; emb_i < n_vocab_; emb_i++)
    {
        Vec_f32x8 dot_accum = { vec_f32x8_setzero() };
        for (int i = 0; i < n_embed; i += 8) {
            Vec_f32x8 x = vec_f32x8_load(inp_data + (ctx_i * n_embed + i));
            Vec_f32x8 w = vec_f32x8_load(weight_data + (emb_i * n_embed + i));
            dot_accum = vec_f32x8_fma(x, w, dot_accum);
        }
        out_data[emb_i] = vec_f32x8_sum(dot_accum);
    }

    return proj_out_;  
}

template<>
Tensor Embedding::forward_proj_impl<Qint8>(const Tensor& inp) {
    // Output probs must be float32.
    Float32* out_data = proj_out_.data_ptr<Float32>();
    const Float32* inp_data = inp.data_ptr<Float32>();
    const Qint8* weight_data = weight_.data_ptr<Qint8>();

    const int ctx_i = inp.size(0) - 1;
    const int n_embed = inp.size(1);

    for (int emb_i = 0; emb_i < n_vocab_; emb_i++)
    {
        Float32 dot_accum = 0.0f;
        for (int i = 0; i < n_embed; i += 1) {
            Float32 x = inp_data[ctx_i * n_embed + i];
            Float32 w = weight_.deq(weight_data[emb_i * n_embed + i]);
            dot_accum += x * w;
        }
        out_data[emb_i] = dot_accum;
    }

    return proj_out_;  
}

Tensor Embedding::forward_proj(const Tensor &inp)
{
    Timer timer(&time_project_ms_);

    GTEN_CHECK_NDIMS_EQUAL(inp.ndims(), 2);
    GTEN_CHECK_DIMSIZE_EQUAL(1, inp.size(1), n_embed_);

    if (weight_.dtype() == kQint8)
        return forward_proj_impl<Qint8>(inp);
    else if (weight_.dtype() == kFloat16)
        return forward_proj_impl<Float16>(inp);
    else
        return forward_proj_impl<Float32>(inp);
}

PosEmbedding::PosEmbedding(const Tensor &weight, AcvConfig acv, int32_t max_ctx)
    : weight_{weight}, max_ctx_{max_ctx},
      n_embed_{weight.size(1)},
      out_{Tensor{{max_ctx, n_embed_}, acv.dtype, acv.scale, acv.zerop}}
{
    GTEN_CHECK_NDIMS_EQUAL(weight_.ndims(), 2);
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

template<>
Tensor PosEmbedding::forward_impl<Qint8>(int32_t n_ctx)
{
    out_.resize({n_ctx, n_embed_});
    Float32* out_data = out_.data_ptr<Float32>();
    const Qint8* weight_data = weight_.data_ptr<Qint8>();

    if (out_cached_) {
        const int data_offset = (n_ctx - 1) * n_embed_;
        for (int i = 0; i < n_embed_; i++)
            out_data[data_offset + i] = weight_.deq(weight_data[data_offset + i]);
    }
    else {
        out_cached_ = true;

        for (int i = 0; i <  n_ctx * n_embed_; i++)
            out_data[i] = weight_.deq(weight_data[i]);
    }
    return out_;
}

Tensor PosEmbedding::forward(int32_t n_ctx)
{
    GTEN_CHECK_INP_CTX_SIZE(n_ctx, max_ctx_);

    Timer timer{&time_ms_};
    
    if (weight_.dtype() == kQint8)
        return forward_impl<Qint8>(n_ctx);
    else if (weight_.dtype() == kFloat16)
        return forward_impl<Float16>(n_ctx);
    else
        return forward_impl<Float32>(n_ctx);
}

LayerNorm::LayerNorm(const Tensor &weight, const Tensor &bias, AcvConfig acv,
                     int32_t max_ctx)
    : weight_{weight}, bias_{bias}, max_ctx_{max_ctx},
      n_embed_{weight.size(0)},
      out_{Tensor{{max_ctx, n_embed_}, acv.dtype, acv.scale, acv.zerop}}
{
    GTEN_CHECK_NDIMS_EQUAL(weight_.ndims(), 1);
    GTEN_CHECK_NDIMS_EQUAL(bias_.ndims(), 1);
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
        for (int i = 0; i < n_embed_; i++)
            mean_accum += fpcvt_to_fp32<scalar_t>(inp_data[i + ctx_offset]);
        Float32 mean = mean_accum / (Float32)n_embed_;

        // Standard deviation calculation.
        Float32 variance_accum = 0.0f;
        for (int i = 0; i < n_embed_; i++) {
            Float32 x = fpcvt_to_fp32<scalar_t>(inp_data[i + ctx_offset]);
            variance_accum += std::pow(x - mean, 2);
        }
        Float32 std_dev = std::sqrt(variance_accum / (Float32)n_embed_);

        // Normalization.
        for (int i = 0; i < n_embed_; i++) {
            Float32 unnormalized = fpcvt_to_fp32<scalar_t>(inp_data[i + ctx_offset]);
            Float32 w = fpcvt_to_fp32<scalar_t>(weight_data[i]);
            Float32 b = fpcvt_to_fp32<scalar_t>(bias_data[i]);
            Float32 normalized = ((unnormalized - mean) / (std_dev + eps_)) * w + b;
            out_data[i + ctx_offset] = fpcvt_from_fp32<scalar_t>(normalized);
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
            for (int i = 0; i < n_embed_; i++)
                mean_accum += fpcvt_to_fp32<scalar_t>(inp_data[i + ctx_offset]);
            Float32 mean = mean_accum / (Float32)n_embed_;

            // Standard deviation calculation.
            Float32 variance_accum = 0.0f;
            for (int i = 0; i < n_embed_; i++) {
                Float32 x = fpcvt_to_fp32<scalar_t>(inp_data[i + ctx_offset]);
                variance_accum += std::pow(x - mean, 2);
            }
            Float32 std_dev = std::sqrt(variance_accum / (Float32)n_embed_);

            // Normalization.
            for (int i = 0; i < n_embed_; i++) {
                Float32 x = fpcvt_to_fp32<scalar_t>(inp_data[i + ctx_offset]);
                Float32 w = fpcvt_to_fp32<scalar_t>(weight_data[i]);
                Float32 b = fpcvt_to_fp32<scalar_t>(bias_data[i]);
                // Epsilon added to standard deviation prevents div by zero.
                Float32 normalized = ((x - mean) / (std_dev + eps_)) * w + b;
                out_data[i + ctx_offset] = fpcvt_from_fp32<scalar_t>(normalized);
            }
        }
    }
    return out_;
}


template<>
Tensor LayerNorm::forward_impl<Qint8>(const Tensor &inp)
{
    out_.resize(inp.shape());
    Float32* out_data = out_.data_ptr<Float32>();
    const Float32* inp_data = inp.data_ptr<Float32>();
    const Qint8* weight_data = weight_.data_ptr<Qint8>();
    const Float32* bias_data = bias_.data_ptr<Float32>();

    const int n_ctx = inp.size(0);

    if (out_cached_)
    {
        const int ctx_offset = (n_ctx - 1) * n_embed_;
        // Mean calculation.
        Float32 mean_accum = 0.0f;
        for (int i = 0; i < n_embed_; i++)
            mean_accum += inp_data[i + ctx_offset];
        Float32 mean = mean_accum / (Float32)n_embed_;

        // Standard deviation calculation.
        Float32 variance_accum = 0.0f;
        for (int i = 0; i < n_embed_; i++) {
            Float32 x = inp_data[i + ctx_offset];
            variance_accum += std::pow(x - mean, 2);
        }
        Float32 std_dev = std::sqrt(variance_accum / (Float32)n_embed_);

        // Normalization.
        for (int i = 0; i < n_embed_; i++) {
            Float32 unnormalized = inp_data[i + ctx_offset];
            Float32 w = weight_.deq(weight_data[i]);
            Float32 b = bias_data[i];
            Float32 normalized = ((unnormalized - mean) / (std_dev + eps_)) * w + b;
            out_data[i + ctx_offset] = normalized;
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
            for (int i = 0; i < n_embed_; i++)
                mean_accum += inp_data[i + ctx_offset];
            Float32 mean = mean_accum / (Float32)n_embed_;

            // Standard deviation calculation.
            Float32 variance_accum = 0.0f;
            for (int i = 0; i < n_embed_; i++) {
                Float32 x = inp_data[i + ctx_offset];
                variance_accum += std::pow(x - mean, 2);
            }
            Float32 std_dev = std::sqrt(variance_accum / (Float32)n_embed_);

            // Normalization.
            for (int i = 0; i < n_embed_; i++) {
                Float32 x = inp_data[i + ctx_offset];
                Float32 w = weight_.deq(weight_data[i]);
                Float32 b = bias_data[i];
                // Epsilon added to standard deviation prevents div by zero.
                Float32 normalized = ((x - mean) / (std_dev + eps_)) * w + b;
                out_data[i + ctx_offset] = normalized;
            }
        }
    }
    return out_;
}

Tensor LayerNorm::forward(const Tensor &inp)
{
    Timer timer(&time_ms_);

    GTEN_CHECK_NDIMS_EQUAL(inp.ndims(), 2);
    GTEN_CHECK_DIMSIZE_EQUAL(1, inp.size(1), n_embed_);

    if (weight_.dtype() == kQint8)
        return forward_impl<Qint8>(inp);
    else if (weight_.dtype() == kFloat16)
        return forward_impl<Float16>(inp);
    else
        return forward_impl<Float32>(inp);
}


GELU::GELU(AcvConfig acv, int32_t max_ctx, int32_t n_out)
    : n_out_{n_out}, out_{Tensor{{max_ctx, n_out_}, acv.dtype, acv.scale, acv.zerop}}
{
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
            Float32 res = 0.5 * x 
                              * (1.0f + std::tanh(std::sqrt(2.0f / 3.141592653589793f)
                              * (x + 0.044715f * std::pow(x, 3.0f))));
            out_data[i] = fpcvt_from_fp32<scalar_t>(res);
        }
    }
    else {
        out_cached_ = true;
        Float32 x;
        for (int i = 0; i < ne; ++i) {
            x = fpcvt_to_fp32<scalar_t>(inp_data[i]);
            Float32 res = 0.5 * x 
                              * (1.0f + std::tanh(std::sqrt(2.0f / 3.141592653589793f)
                              * (x + 0.044715f * std::pow(x, 3.0f))));
            out_data[i] = fpcvt_from_fp32<scalar_t>(res);
        }
    }
    return out_;
}


Tensor GELU::forward(const Tensor& inp)
{
    Timer timer{&time_ms_};

    GTEN_CHECK_NDIMS_EQUAL(inp.ndims(), 2);
    GTEN_CHECK_DIMSIZE_EQUAL(1, inp.size(1), n_out_);

    if (out_.dtype() == kQint8)
        return forward_impl<Float32>(inp);
    else if (out_.dtype() == kFloat16)
        return forward_impl<Float16>(inp);
    else
        return forward_impl<Float32>(inp);
}

Residual::Residual(AcvConfig acv, int32_t max_ctx, int32_t n_embed)
    : out_{Tensor{{max_ctx, n_embed}, acv.dtype, acv.scale, acv.zerop}}
{
}

Tensor Residual::forward(const Tensor& inp0, const Tensor& inp1)
{
    Timer timer{&time_ms_};

    GTEN_CHECK_DTYPE_EQUAL(inp0.dtype(), inp1.dtype());
    GTEN_CHECK_NDIMS_EQUAL(inp0.ndims(), 2);
    GTEN_CHECK_NDIMS_EQUAL(inp1.ndims(), 2);
    GTEN_ASSERT(
        inp0.shape() == inp1.shape(),
        "Expected input tensor1 shape=(%d, %d) to be equal to tensor2 shape=(%d, %d).",
        inp0.size(0), inp0.size(1), inp1.size(0), inp1.size(1));

    if (out_.dtype() == kQint8)
        return forward_impl<Float32>(inp0, inp1);
    else if (out_.dtype() == kFloat16)
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

    if (out_cached_)
    {
        uint32_t n_iter = n_embed;
        uint32_t offset = inp0.numel() - n_embed;
        for (uint32_t i = 0; i < n_iter; i += 8) {
            Vec_f32x8 x0 = vec_f32x8_load(inp0_data + offset + i);
            Vec_f32x8 x1 = vec_f32x8_load(inp1_data + offset + i);
            Vec_f32x8 x_sum = vec_f32x8_add(x0, x1);
            vec_f32x8_store(x_sum, out_data + offset + i);
        }
    }
    else {
        out_cached_ = true;
        int n_iter = inp0.numel();
        for (int i = 0; i < n_iter; i += 8) {
            Vec_f32x8 x0 = vec_f32x8_load(inp0_data + i);
            Vec_f32x8 x1 = vec_f32x8_load(inp1_data + i);
            Vec_f32x8 x_sum = vec_f32x8_add(x0, x1);
            vec_f32x8_store(x_sum, out_data + i);
        }
    }

    return out_;
}

Linear::Linear(const Tensor &weight, const Tensor &bias, AcvConfig acv,
               int32_t max_ctx)
    : weight_{weight}, bias_{bias},
      out_{Tensor{{max_ctx, weight_.size(0)}, acv.dtype, acv.scale, acv.zerop}}
{
    GTEN_CHECK_NDIMS_EQUAL(weight_.ndims(), 2);
    GTEN_CHECK_NDIMS_EQUAL(bias_.ndims(), 1);
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

    if (out_cached_)
    {
        const int ctx_i = n_ctx - 1;
        for (int out_i = 0; out_i < n_out; out_i++) {
            Vec_f32x8 dot_accum = vec_f32x8_setzero();
            for (int i = 0; i < n_embed; i += 8) {
                Vec_f32x8 x0 = vec_f32x8_load(inp_data + (ctx_i * n_embed + i));
                Vec_f32x8 x1 = vec_f32x8_load(weight_data + (out_i * n_embed + i));
                dot_accum = vec_f32x8_fma(x0, x1, dot_accum);
            }
            Float32 bias = fpcvt_to_fp32<scalar_t>(bias_data[out_i]);
            Float32 res =  vec_f32x8_sum(dot_accum) + bias;
            out_data[ctx_i * n_out + out_i] = fpcvt_from_fp32<scalar_t>(res);
        }
    }
    else
    {
        out_cached_ = true;
        for (int ctx_i = 0; ctx_i < n_ctx; ctx_i++) {
            for (int out_i = 0; out_i < n_out; out_i++) {
                Vec_f32x8 dot_accum = vec_f32x8_setzero();
                for (int i = 0; i < n_embed; i += 8) {
                    Vec_f32x8 x0 = vec_f32x8_load(inp_data + (ctx_i * n_embed + i));
                    Vec_f32x8 x1 = vec_f32x8_load(weight_data + (out_i * n_embed + i));
                    dot_accum = vec_f32x8_fma(x0, x1, dot_accum);
                }
                Float32 bias = fpcvt_to_fp32<scalar_t>(bias_data[out_i]);
                Float32 res =  vec_f32x8_sum(dot_accum) + bias;
                out_data[ctx_i * n_out + out_i] = fpcvt_from_fp32<scalar_t>(res);
            }
        }
    }

    return out_;
}


template<>
Tensor Linear::forward_impl<Qint8>(const Tensor& inp)
{
    const int n_ctx = inp.size(0);
    const int n_embed = inp.size(1);
    const int n_out = weight_.size(0);
    
    out_.resize({n_ctx, n_out});
    Float32* out_data = out_.data_ptr<Float32>();
    const Qint8* weight_data = weight_.data_ptr<Qint8>();
    const Float32* bias_data = bias_.data_ptr<Float32>();
    const Float32* inp_data = inp.data_ptr<Float32>();

    if (out_cached_)
    {
        const int ctx_i = n_ctx - 1;
        for (int out_i = 0; out_i < n_out; out_i++) {
            Float32 dot_accum = 0.0f;
            for (int i = 0; i < n_embed; i += 1) {
                Float32 x0 = inp_data[ctx_i * n_embed + i];
                Float32 x1 = weight_.deq(weight_data[out_i * n_embed + i]);
                dot_accum += x0 * x1;
            }
            Float32 bias = bias_data[out_i];
            Float32 res =  dot_accum + bias;
            out_data[ctx_i * n_out + out_i] = res;
        }
    }
    else
    {
        out_cached_ = true;
        for (int ctx_i = 0; ctx_i < n_ctx; ctx_i++) {
            for (int out_i = 0; out_i < n_out; out_i++) {
                Float32 dot_accum = 0.0f;
                for (int i = 0; i < n_embed; i += 1) {
                    Float32 x0 = inp_data[ctx_i * n_embed + i];
                    Float32 x1 = weight_.deq(weight_data[out_i * n_embed + i]);
                    dot_accum += x0 * x1;
                }
                Float32 bias = bias_data[out_i];
                Float32 res =  dot_accum + bias;
                out_data[ctx_i * n_out + out_i] = res;
            }
        }
    }

    return out_;
}

Tensor Linear::forward(const Tensor &inp)
{
    Timer timer{&time_ms_};

    GTEN_CHECK_NDIMS_EQUAL(inp.ndims(), 2);
    GTEN_CHECK_DIMSIZE_EQUAL(1, inp.size(1), weight_.size(1));

    if (weight_.dtype() == kQint8)
        return forward_impl<Qint8>(inp);
    else if (weight_.dtype() == kFloat16)
        return forward_impl<Float16>(inp);
    else
        return forward_impl<Float32>(inp);
}

MultiHeadSelfAttn::MultiHeadSelfAttn(const Linear& query, const Linear& key,
                                     const Linear& value, const Linear& out_proj,
                                     AcvConfig acv, int32_t max_ctx,
                                     int32_t n_embed, int32_t n_head)
    : query_{query}, key_{key}, value_{value}, qkv_proj_{out_proj}, n_head_{n_head},
      qkv_cache_{Tensor{{max_ctx, n_embed}, acv.dtype, acv.scale, acv.zerop}}
{
    if (acv.dtype == kFloat16) {
        qk_cache_ = Tensor({n_head, max_ctx, max_ctx}, kFloat16);
        Float16 *qk_cache_data = qk_cache_.data_ptr<Float16>();
        const int ne = qk_cache_.numel();
        const Float16 inf = fp32_to_fp16(-INFINITY);
        for (int i = 0; i < ne; i++)
            qk_cache_data[i] = inf;
    }
    else {
        qk_cache_ = Tensor({n_head, max_ctx, max_ctx}, kFloat32);
        Float32 *qk_cache_data = qk_cache_.data_ptr<Float32>();
        const int ne = qk_cache_.numel();
        for (int i = 0; i < ne; i++)
            qk_cache_data[i] = -INFINITY;
    }
}

Tensor MultiHeadSelfAttn::forward(const Tensor &inp)
{
    GTEN_CHECK_NDIMS_EQUAL(inp.ndims(), 2);
    GTEN_CHECK_DIMSIZE_EQUAL(1, inp.size(1), qkv_cache_.size(1));

    const int n_ctx = inp.size(0);
    const int n_embed = inp.size(1);

    Tensor q = query_.forward(inp);
    Tensor k = key_.forward(inp);
    Tensor v = value_.forward(inp);

    const Tensor qkv = (q.dtype() == kFloat16)
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
                qk_data[qk_data_i] = fpcvt_from_fp32<scalar_t>(dot_prod);
            }
        }

        // SOFTMAX
        for (int head = 0; head < n_head_; head++)
        {
            const int row = n_ctx - 1;
            const int base_i = head * n_ctx * n_ctx + row * n_ctx;

            Float32 max = -INFINITY;
            for (int i = 0; i < n_ctx; i++) {
                Float32 val = fpcvt_to_fp32<scalar_t>(qk_data[base_i + i]);
                if (val > max)
                    max = val;
            }

            Float32 sum_exp = 0;
            for (int i = 0; i < n_ctx; i++) {
                int qk_i = base_i + i;
                Float32 val = fpcvt_to_fp32<scalar_t>(qk_data[qk_i]);
                qk_data[qk_i] = fpcvt_from_fp32<scalar_t>(std::exp(val - max));
                sum_exp += fpcvt_to_fp32<scalar_t>(qk_data[qk_i]);
            }

            for (int i = 0; i < n_ctx; i++) {
                Float32 qkw = fpcvt_to_fp32<scalar_t>(qk_data[base_i + i]);
                qk_data[base_i + i] = fpcvt_from_fp32<scalar_t>(qkw / sum_exp);
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
                    Float32 qkw = fpcvt_to_fp32<scalar_t>(qk_data[qk_i]);
                    Float32 vw = fpcvt_to_fp32<scalar_t>(v_data[v_i]);
                    dot_prod += qkw * vw;
                }
                int qkv_data_i = head * d_head + qk_row * n_embed + v_col;
                qkv_data[qkv_data_i] = fpcvt_from_fp32<scalar_t>(dot_prod);
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
                        Float32 qw = fpcvt_to_fp32<scalar_t>(q_data[q_i]);
                        Float32 kw = fpcvt_to_fp32<scalar_t>(k_data[k_i]);
                        dot_accum += qw * kw * scale_factor;
                    }
                    int qk_data_i = head * n_ctx * n_ctx + q_row * n_ctx + k_col;
                    qk_data[qk_data_i] = fpcvt_from_fp32<scalar_t>(dot_accum);
                }

            }
        }

        // SOFTMAX
        int n_rows = n_head_ * n_ctx;
        for (int row = 0; row < n_rows; row++)
        {
            Float32 max = -INFINITY;
            for (int i = 0; i < n_ctx; i++) {
                Float32 x = fpcvt_to_fp32<scalar_t>(qk_data[row * n_ctx + i]);
                if (x > max)
                    max = x;
            }

            Float32 sum_exp = 0;
            for (int i = 0; i < n_ctx; i++) {
                int qk_i = row * n_ctx + i;
                Float32 x = fpcvt_to_fp32<scalar_t>(qk_data[qk_i]);
                qk_data[qk_i] = fpcvt_from_fp32<scalar_t>(std::exp(x - max));
                sum_exp += fpcvt_to_fp32<scalar_t>(qk_data[qk_i]);
            }

            for (int i = 0; i < n_ctx; i++) {
                Float32 qkw =  fpcvt_to_fp32<scalar_t>(qk_data[row * n_ctx + i]);
                qk_data[row * n_ctx + i] = fpcvt_from_fp32<scalar_t>(qkw / sum_exp);
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
                        Float32 qkw =fpcvt_to_fp32<scalar_t>(qk_data[qk_i]);
                        Float32 vw = fpcvt_to_fp32<scalar_t>(v_data[v_i]);
                        dot_prod += qkw * vw;
                    }
                    int qkv_data_i = head * d_head + qk_row * n_embed + v_col;
                    qkv_data[qkv_data_i] = fpcvt_from_fp32<scalar_t>(dot_prod);
                }
            }
        }
    }

    return qkv_cache_;
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
