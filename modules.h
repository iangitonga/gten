#pragma once


#include "tensor.h"

#include <chrono>
#include <iostream>

namespace gten
{

class Timer
{
public:
    Timer(int64_t *time_tracker) : time_tracker_(time_tracker) { start_time_ = std::chrono::high_resolution_clock::now(); }
    ~Timer() { stop(); }

    void stop() {
        if (stopped_)
            return;
        auto end_time = std::chrono::high_resolution_clock::now();
        int64_t start = std::chrono::time_point_cast<std::chrono::milliseconds>(start_time_).time_since_epoch().count();
        int64_t end = std::chrono::time_point_cast<std::chrono::milliseconds>(end_time).time_since_epoch().count();
        int64_t duration = end - start;
        *time_tracker_ += duration;
        stopped_ = true;
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    int64_t *time_tracker_;
    bool stopped_ = false;
};

/// Provides an embedding table lookup for tokens.
class Embedding
{
public:
    Embedding() {}

    /// Construct an embedding table from the given weight. The weight should be of shape
    /// (n_vocab, n_embed).
    Embedding(const Tensor &weight, uint32_t max_ctx);

    /// Returns the embeddings of the given tokens. The input tensor must be of shape
    /// (n_ctx,) and the output tensor is of shape (n_ctx, n_embed).
    Tensor forward(const Tensor& tokens);

    /// Performs matrix mul between the input and the weights tensor. The input tensor
    /// is expected to have shape (n_ctx, n_embed) and the output has shape (1, n_vocab)
    /// which represents the prob dist of the next predicted token given the context.
    Tensor forward_project(const Tensor &inp);
    int64_t emb_time() const { return time_embed_ms_; }
    int64_t emb_proj_time() const { return time_project_ms_; }

private:
    uint32_t n_vocab_;
    uint32_t n_embed_;
    uint32_t max_ctx_;
    Tensor weight_;
    Tensor emb_out_;
    Tensor proj_out_;
    bool emb_out_cached_ = false;
    bool proj_out_cached_ = false;

    int64_t time_embed_ms_ = 0;
    int64_t time_project_ms_ = 0;
};

/// Provides a positional embedding table lookup for tokens.
class PosEmbedding
{
public:
    PosEmbedding() {}

    /// Construct a positional embedding table from the given weight. The weight should be
    /// of shape (max_ctx, n_embed).
    PosEmbedding(const Tensor &weight, uint32_t max_ctx);

    /// Return the positional embeddings of the given number of tokens. The number of
    /// tokens must not exceed max_ctx.
    Tensor forward(uint32_t n_ctx);
    int64_t time() const { return time_ms_; }
    
private:
    uint32_t max_ctx_;
    uint32_t n_embed_;
    Tensor weight_;

    Tensor out_;
    bool out_cached_;
    int64_t time_ms_ = 0;
};

class LayerNorm
{
public:
    LayerNorm() {}

    /// Construct affine layer-norm from the given weight and bias. The weight and bias
    /// should be of shape (n_embed,).
    LayerNorm(const Tensor &weight, const Tensor &bias, uint32_t max_ctx);

    /// Normalize the input. Both input and output are of shape (n_ctx, n_embed). n_ctx
    /// must not exceed max_ctx.
    Tensor forward(const Tensor &inp);
    int64_t time() const { return time_ms_; }

private:
    uint32_t n_embed_;
    Tensor weight_;
    Tensor bias_;
    uint32_t max_ctx_;
    float eps_ = 1e-05f;

    Tensor out_;
    bool out_cached_ = false;
    int64_t time_ms_ = 0;
};

class Residual
{
public:
    Residual() {}
    Residual(uint32_t max_ctx, uint32_t n_embed);
    Tensor forward(const Tensor &inp0, const Tensor &inp1);
    int64_t time() const { return time_ms_; }

private:
    Tensor out_;
    bool out_cached_ = false;
    int64_t time_ms_ = 0;
};

class GELU
{
public:
    GELU() {};
    GELU(uint32_t max_ctx, uint32_t n_out);
    Tensor forward(const Tensor &inp);
    int64_t time() const { return time_ms_; }

private:
    uint32_t n_out_;

    Tensor out_;
    bool out_cached_ = false;
    int64_t time_ms_ = 0;
};


class Linear
{
public:
    Linear() {}

    /// Construct linear layer from the given weight and bias. The weight should be of
    /// shape (n_out, n_embed) and bias (n_out).
    Linear(const Tensor &weight, const Tensor &bias, uint32_t max_ctx);
    Tensor forward(const Tensor &inp);
    int64_t time() const { return time_ms_; }

private:
    Tensor weight_;
    Tensor bias_;
    Tensor out_;
    bool out_cached_ = false;
    int64_t time_ms_ = 0;
};

class MultiHeadSelfAttn
{
public:
    MultiHeadSelfAttn() {}

    /// Construct self-attn layer from the given weights. query, key and value should be
    /// of shape (n_embed, n_head x d_head). out_proj of shape (n_head x d_head, n_embed).
    MultiHeadSelfAttn(const Linear &query, const Linear &key, const Linear &value, const Linear &out_proj,
                      uint32_t max_ctx, uint32_t n_embed, uint32_t n_head);
    Tensor forward(const Tensor &inp);
    int64_t time_linear() const { return query_.time() + key_.time() + value_.time() + out_proj_.time(); }
    int64_t time_attn() const { return time_attn_ms_; }

private:
    Linear query_;
    Linear key_;
    Linear value_;
    Linear out_proj_;
    uint32_t n_head_;

    Tensor qk_cache_;
    Tensor qkv_cache_;
    bool qkv_cached_ = false;
    int64_t time_attn_ms_ = 0;

    Tensor qkv_attn(const Tensor &q, const Tensor &k, const Tensor &v);
};


class ResidualAttentionBlock
{
public:
    ResidualAttentionBlock() {}
    ResidualAttentionBlock(const MultiHeadSelfAttn &attn, const LayerNorm &ln_1, const Linear &mlp_fc,
                           const Linear &mlp_proj, const LayerNorm &ln_2, const GELU &gelu,
                           uint32_t max_ctx, uint32_t n_embed);
    Tensor forward(const Tensor &inp);
    int64_t time_linear() const { return attn_.time_linear() + mlp_fc_.time() + mlp_proj_.time(); }
    int64_t time_proj() const { return mlp_fc_.time() + mlp_proj_.time(); }
    int64_t time_attn_lin() const { return attn_.time_linear(); }
    int64_t time_attn() const { return attn_.time_attn(); }
    int64_t time_ln() const { return ln_1_.time() + ln_2_.time(); }
    int64_t time_gelu() const { return gelu_.time(); }
    int64_t time_res() const { return inp_res_.time() + attn_res_.time(); }

private:
    MultiHeadSelfAttn attn_;
    LayerNorm ln_1_;
    Linear mlp_fc_;
    Linear mlp_proj_;
    LayerNorm ln_2_;
    GELU gelu_;
    Residual inp_res_;
    Residual attn_res_;
};

} // namespace gten

