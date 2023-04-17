#pragma once


#include "tensor.h"
#include "tokenizer.h"

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

class Embedding
{
public:
    Embedding() {}
    Embedding(const Tensor &weight, const uint32_t n_vocab, const uint32_t max_ctx, const uint32_t n_embed);
    Tensor forward(const Tensor& tokens);
    Tensor forward_project(const Tensor &inp);
    int64_t emb_time() const { return time_embed_ms_; }
    int64_t emb_proj_time() const { return time_project_ms_; }

private:
    uint32_t n_vocab_;
    uint32_t n_embed_;
    Tensor weight_;
    Tensor emb_out_;
    Tensor proj_out_;
    bool emb_out_cached_ = false;
    bool proj_out_cached_ = false;

    int64_t time_embed_ms_ = 0;
    int64_t time_project_ms_ = 0;
};


class PosEmbedding
{
public:
    PosEmbedding() {}
    PosEmbedding(const Tensor &weight, const uint32_t max_ctx, const uint32_t n_embed);
    Tensor forward(const uint32_t n_ctx);
    int64_t time() const { return time_ms_; }
    
private:
    uint32_t max_ctx_;
    uint32_t n_embed_;
    Tensor weight_;

    Tensor out_;
    bool out_cached_;
    int64_t time_ms_ = 0;
};

// Normalizes each vector in the given sequence of vector independently.
class LayerNorm
{
public:
    LayerNorm() {}
    LayerNorm(const Tensor &weight, const Tensor &bias, const uint32_t max_ctx, const uint32_t n_embed);
    Tensor forward(const Tensor &inp);
    int64_t time() const { return time_ms_; }

private:
    uint32_t n_embed_;
    // A float tensor of shape (n_embed,)
    Tensor weight_;
    // A float tensor of shape (n_embed,)
    Tensor bias_;
    float eps_ = 1e-05f;

    Tensor out_;
    bool out_cached_ = false;
    int64_t time_ms_ = 0;
};

class Residual
{
public:
    Residual() {}
    Residual(const uint32_t max_ctx, const uint32_t n_embed);
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
    GELU(const uint32_t max_ctx, const uint32_t n_out);
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
    Linear(const Tensor &weight, const Tensor &bias, const uint32_t max_ctx);
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
    MultiHeadSelfAttn(const Linear &query, const Linear &key, const Linear &value, const Linear &out_proj,
                      const uint32_t max_ctx, const uint32_t n_embed);
    Tensor forward(const Tensor &inp);
    int64_t time_linear() const { return query_.time() + key_.time() + value_.time() + out_proj_.time(); }
    int64_t time_attn() const { return time_attn_ms_; }

private:
    Linear query_;
    Linear key_;
    Linear value_;
    Linear out_proj_;

    Tensor qk_cache_;
    Tensor qkv_cache_;
    bool qkv_cached_ = false;
    int64_t time_attn_ms_ = 0;

    Tensor qkv_attn(const Tensor &q, const Tensor &k, const Tensor &v, const uint32_t n_head);
};


class ResidualAttentionBlock
{
public:
    ResidualAttentionBlock() {}
    ResidualAttentionBlock(const MultiHeadSelfAttn &attn, const LayerNorm &ln_1, const Linear &mlp_fc,
                           const Linear &mlp_proj, const LayerNorm &ln_2, const GELU &gelu,
                           const uint32_t max_ctx, const uint32_t n_embed);
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


struct GPT2Config
{
    int64_t n_vocab;
    int64_t n_ctx;
    int64_t n_embed;
    int64_t n_layer;
    int64_t n_head;

    friend std::ostream& operator<<(std::ostream& stream, const GPT2Config& config)
    {
        stream << "GPT2Config(n_vocab=" << config.n_vocab
               << ", n_ctx="   << config.n_ctx
               << ", n_embed=" << config.n_embed
               << ", n_layer=" << config.n_layer
               << ", n_head="  << config.n_head
               << ")\n";
        return stream;
    }
};


class GPT2
{
public:
    GPT2Tokenizer tokenizer;

    GPT2(const std::string &fname);
    Tensor logits(const Tensor &inp);
    void show_performance() const;
    void sample(const std::string &prompt, double temp = 1.0, int max_iter = 1024);

private:
    const int64_t magic_number_ = 0x454c49464e455447;
    GPT2Config config_;

    Embedding wte_;
    PosEmbedding wpe_;
    std::vector<ResidualAttentionBlock> blocks_;
    LayerNorm ln_f_;
    Residual res_;

    int64_t time_sample_ms_ = 0;
    int64_t niter_ = 0;
    
    void load_from_file(const std::string &fname);
};

} // namespace gten

