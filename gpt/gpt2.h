#include "gten.h"
#include "tokenizer.h"


struct InferenceOptions {
    std::string model_name {"Gpt2-medium"};
    std::string prompt {""};
    int gen_tokens {200}; // number of tokens to generate.
    float temp {0.9f};
    bool debug_mode {false};
    bool greedy {false};

    std::string get_dl_command() const {
        #if defined(__WIN32__) || defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
        return std::string("python model_dl.py ") + model_name;
        #else
        return std::string("python3 model_dl.py ") + model_name;
        #endif
    }

    std::string get_model_path() const {
        return std::string("models/") + model_name + ".fp16.gten";
    }

    void print_debug_info() const {
        if (debug_mode) {
            std::cout << "Model name     : " << model_name << "\n";
            std::cout << "Model path     : " << get_model_path() << "\n";
            std::cout << "Inference      : " << "FP16" << "\n";
            std::cout << "Temperature    : " << temp << "\n";
            std::cout << "Tokens to gen  : " << gen_tokens << "\n";
        }
    }

    int calculate_max_ctx_size(int num_prompt_tokens) const {
        // max ctx_size for gpt2 models.
        int max_ctx_size = 1024;

        if (num_prompt_tokens >= max_ctx_size) {
            // Prompt length is too large, quit. Technically, we can allow generation of
            // arbitrary-length documents by selecting the last 1000 context tokens and using
            // that to predict the next token but the modules are not yet designed with that
            // in mind. In the future that feature will be available.
            GTEN_ASSERT(false, "Prompt length is too large!");
        }
        // How many tokens: gen_tokens + prompt tokens
        int ctx_size = num_prompt_tokens + gen_tokens;

        // Round of to the nearest power of two.
        if (ctx_size < 32)
            return 32;
        else if (ctx_size < 64)
            return 64;
        else if (ctx_size < 128)
            return 128;
        else if (ctx_size < 256)
            return 256;
        else if (ctx_size < 512)
            return 512;
        else if (ctx_size < 768)
            return 768;
        else
            return max_ctx_size;
    }

};

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
    gten::GPT2Tokenizer tokenizer_;

    GPT2(const GPT2Config& config, int max_ctx, gten::GPT2Tokenizer&& tokenizer);
    gten::Tensor logits(const gten::Tensor &inp);
    void show_performance(int64_t niter) const;
    void sample(const InferenceOptions& opts);
    void greedy_sample(const InferenceOptions& opts);
    void reset_acv_caches();

public:
    GPT2Config config_;
    gten::Embedding wte_;
    gten::PosEmbedding wpe_;
    std::vector<gten::ResidualAttnBlock> blocks_;
    gten::LayerNorm ln_f_;
    gten::Residual res_;
    int64_t time_sample_ms_ = 0;
    int64_t time_load_ms_ = 0;
};


GPT2* load_gpt(const InferenceOptions &inference_opts, bool show_load_info);
