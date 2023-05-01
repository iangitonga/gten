#include "modules.h"
#include "tokenizer.h"


struct GPT2Config
{
    int32_t n_vocab, n_ctx, n_embed, n_layer, n_head;

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
    gten::GPT2Tokenizer tokenizer;

    GPT2(const std::string &fname);
    gten::Tensor logits(const gten::Tensor &inp);
    void show_performance(int64_t niter) const;
    void sample(const std::string &prompt, double temp = 1.0, int max_iter = 1000);
    void greedy_sample(const std::string &prompt, double temp, int max_iter);

private:
    const int64_t magic_number_ = 0x454c49464e455447;
    GPT2Config config_;
    gten::Embedding wte_;
    gten::PosEmbedding wpe_;
    std::vector<gten::ResidualAttentionBlock> blocks_;
    gten::LayerNorm ln_f_;
    gten::Residual res_;
    int64_t time_sample_ms_ = 0;
    int64_t time_load_ms_ = 0;
    
    void load_from_file(const std::string &fname);
};
