#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <random>

#include "gpt2.h"


using namespace gten;


const char *usage = R"(
usage:
gpt2 [options] -p PROMPT  for a single prompt or
gpt2 [options] for a chat interface. 

Optional args.
-sm :      Use small model (117M) for inference.
-md :      Use medium model (345M) for inference. This model is chosen by default.
-lg :      Use large model (762M) for inference.
-debug   : See debug-level information.
--temp T : Temperature to use during sampling. It must be greater than 0. [default=0.9].
--len  L : Number of words to generate. Minimum is 1 and max is 1000. [default=200].

Examples:
  ./gpt2 -p "Once upon a time" 
  ./gpt2 -lg -p "Once upon a time"
  ./gpt2 -lg --temp 0.5 -p "Once upon a time"
  ./gpt2
)";


int main(int argc, char const *argv[])
{
    if (argc < 2) {
        std::cout << "Prompt is not provided.\n";
		std::cout << usage << "\n";
		return -1;
    }
    
    InferenceOptions options{};

    for (int i = 1; i < argc; i++)
    {
        std::string_view arg(argv[i]);
        if (arg == "-h" || arg == "--help") {
            std::cout << usage << "\n";
            return -1;
        }
        else if (arg == "-p") {
            if (argc <= i+1) {
                std::cout << "Prompt is missing.\n";
                return -1;
            }
            options.prompt = argv[i+1];
            i += 1;
        }
        else if (arg == "-sm") {
            options.model_name = "Gpt2";
        }
        else if (arg == "-md") {
            options.model_name = "Gpt2-medium";
        }
        else if (arg == "-lg") {
            options.model_name = "Gpt2-large";
        }
        else if (arg == "-debug") {
            options.debug_mode = true;
        }
        else if (arg == "-greedy") {
            options.greedy = true;
        }
        else if (arg == "--temp") {
            if (argc <= i+1) {
                std::cout << "Temp value is missing.\n";
                return -1;
            }
            float temp;
            try {
                temp = std::stof(argv[i+1]);
            } catch (...) {
                std::cout << "Invalid temp value.\n";
                return -1;
            }
            if (temp <= 0.0f) {
                std::cout << "Temp value must be greater than zero.\n";
                return -1;
            }
            options.temp = temp;
            i += 1; // skip parsed temp.
        }
        else if (arg == "--len") {
            if (argc <= i+1) {
                std::cout << "Length value is missing.\n";
                return -1;
            }
            int len;
            try {
                len = std::stoi(argv[i+1]);
            } catch (...) {
                std::cout << "Invalid Length value.\n";
                return -1;
            }
            if (len < 1 || len > 1024) {
                std::cout << "Length must be greater than 1 and less than 1000.\n";
                return -1;
            }
            options.gen_tokens = len;
            i += 1;
        }
        else {
            std::cout << "Unknown option: " << arg << "\n";
            return -1;
        }
    }

    options.print_debug_info();

    int res = std::system(options.get_dl_command().c_str());
    if (res != 0) {
        std::cout << "Error: Failed to download " << options.model_name << " model due to network issues.\n";
        return -1;
    }

    GPT2* model = load_gpt(options, false);

    if (options.prompt == "") {
        std::cout << "Chat interface. Write your prompt and press enter to submit. Enter q or press ctrl+c to quit.\n";
        std::string prompt;
        while (true) {
            std::cout << "\n\n\x1B[0m[You]: ";
            std::getline(std::cin, prompt);
            if (prompt == "q")
                break;

            options.prompt = prompt;

            if (options.greedy)
                model->greedy_sample(options);
            else
                model->sample(options);

            model->reset_acv_caches();
        }
    }
    else {
        if (options.greedy)
            model->greedy_sample(options);
        else
            model->sample(options);
    }

    return 0;
}

gten::Tensor GPT2::logits(const gten::Tensor &inp)
{
    gten::Tensor logits = res_.forward(wte_.forward(inp), wpe_.forward(inp.size(0)));
    for (auto &block : blocks_)
        logits = block.forward(logits);
    logits = ln_f_.forward(logits);
    logits = wte_.forward_proj(logits);

    return logits;
}

void GPT2::reset_acv_caches() {
    res_.reset_acv_cache();
    wte_.reset_acv_cache();
    for (auto &block : blocks_)
        block.reset_acv_cache();
    ln_f_.reset_acv_cache();
}


// Used for debugging purposes.
void GPT2::greedy_sample(const InferenceOptions& opts)
{
    time_sample_ms_ = 0;

    const int max_ctx_size = 128;

    std::vector<int32_t> tokens = tokenizer_.encode(opts.prompt);
    tokens.reserve(max_ctx_size);
    gten::Tensor logits;
    const int logits_size = 50257;

    const int eot_token = 50256;
    const int initial_pos = tokens.size() - 1;
    const int n_iter = max_ctx_size;
    int64_t niter = 0;
    // Use cerr because it is unbuffered.
    std::cerr << "\n\n" << opts.prompt;
    for (int i = initial_pos; i < n_iter; i++)
    {
        // TODO: allow creation of tensors with external non-owning data.
        gten::Tensor input(tokens.data(), {(int32_t)tokens.size()}, gten::kInt32);
        gten::Tensor logits = this->logits(input);

        gten::Timer timer(&time_sample_ms_);
        const float *logits_data = logits.data_ptr<float>();

        float max_prob = -INFINITY;
        int max_index = 0;
        for (int j = 0; j < logits_size; ++j){
            if (logits_data[j] > max_prob) {
                max_prob = logits_data[j];
                max_index = j;
            }
        }

        int maxi = max_index;
        if (maxi == eot_token)
            break;
        std::cerr << tokenizer_.decode(maxi);
        tokens.push_back(maxi);

        niter += 1;
    }
    std::cerr << "\n";

    show_performance(niter);
}

void GPT2::sample(const InferenceOptions& opts)
{
    time_sample_ms_ = 0;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<int32_t> tokens = tokenizer_.encode(opts.prompt);
    const int max_ctx_size = opts.calculate_max_ctx_size(tokens.size());
    tokens.reserve(max_ctx_size);
    const int logits_size = 50257;
    std::vector<std::pair<double, int>> logits_probs;
    logits_probs.reserve(logits_size);
    const int eot_token = 50256;
    const int initial_pos = tokens.size();

    // Total ntokens = Requested number of tokens + prompt num of tokens.
    // int total_ntokens = opts.gen_tokens + tokens.size();
    int total_ntokens = opts.gen_tokens;
    // If the total_ntokens > max_prompt_size, generate up to
    // max_prompt_size. Else generate up to requested size.
    const int max_iter = total_ntokens > 1000 ? 1000 : total_ntokens;
    // std::cout << "Mi=" << max_iter << ", in=" << initial_pos << "\n";
	int64_t niter = 0;
    // Use cerr because it is unbuffered.
    std::cerr << "\n[GPT]: \n\n";
    std::cerr << opts.prompt << "\x1B[1;34m"; 
    for (int i = initial_pos; i < max_iter; i++)
    {
        // TODO: allow creation of tensors with external non-owning data.
        gten::Tensor input{(void*)tokens.data(), {(int32_t)tokens.size()}, gten::kInt32};
        gten::Tensor logits = this->logits(input);

        gten::Timer timer(&time_sample_ms_);
        const float *logits_data = logits.data_ptr<float>();

        logits_probs.clear();
        for (int j = 0; j < logits_size; ++j)
            logits_probs.push_back(std::make_pair((double)logits_data[j] / opts.temp, j));

        const int top_k = 40;
        
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
        std::cerr << tokenizer_.decode(maxi);
        tokens.push_back(maxi);

        niter += 1;
    }
    std::cerr << "\n";

    // if (opts.debug_mode)
	show_performance(niter);
}


void GPT2::show_performance(int64_t niter) const
{
    if (niter < 1)
        return;

    int64_t emb_time = wte_.emb_time();
    int64_t emb_proj_time = wte_.emb_proj_time();
    int64_t wpe_time = wpe_.time();
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
    int64_t total = emb_time + emb_proj_time + wpe_time + linear_time + attn_time
                    + ln_time + gelu_time + res_time + time_sample_ms_;

    std::cout << "\n";
    std::cout << "--------------------------------------\n";
    std::cout << "LAYER/OP    TIME PER TOKEN  TIME TOTAL\n";
    std::cout << "--------------------------------------\n";
    std::cout << "Embedding      | " << std::setw(3) << emb_time/niter        << "ms | " << emb_time        << "ms\n";
    std::cout << "Embedding proj | " << std::setw(3) << emb_proj_time/niter   << "ms | " << emb_proj_time   << "ms\n";
    std::cout << "Pos embedding  | " << std::setw(3) << wpe_time/niter        << "ms | " << wpe_time        << "ms\n";
    std::cout << "Linear(qkv+mlp)| " << std::setw(3) << linear_time/niter     << "ms | " << linear_time     << "ms\n";
    // std::cout << "Linear (qkv)   | " << std::setw(2) << attn_lin/niter     << "ms | " << attn_lin        << "ms\n";
    // std::cout << "Linear (mlp)   | " << std::setw(2) << mlpp_time/niter    << "ms | " << mlpp_time       << "ms\n";
    std::cout << "Attention      | " << std::setw(3) << attn_time/niter       << "ms | " << attn_time       << "ms\n";
    std::cout << "Layer norm     | " << std::setw(3) << ln_time/niter         << "ms | " << ln_time         << "ms\n";
    std::cout << "Gelu           | " << std::setw(3) << gelu_time/niter       << "ms | " << gelu_time       << "ms\n";
    std::cout << "Residual       | " << std::setw(3) << res_time/niter        << "ms | " << res_time        << "ms\n";
    std::cout << "Sampler        | " << std::setw(3) << time_sample_ms_/niter << "ms | " << time_sample_ms_ << "ms\n";
    std::cout << "Loading        | " << std::setw(3) << ""                    << "   | " << time_load_ms_   << "ms\n";
    std::cout << "--------------------------------------\n";
    std::cout << "TOTAL          | " << std::setw(3) << total/niter    << "ms | " << total << "ms\n";
    std::cout << "--------------------------------------\n";
}

static inline void read_block_header(std::ifstream& fin, bool debug = false)
{
    std::string block_name;
    int32_t block_name_size;
    fin.read(reinterpret_cast<char*>(&block_name_size), sizeof(block_name_size));
    block_name.resize(block_name_size);
    fin.read(reinterpret_cast<char*>(block_name.data()), block_name_size);

    // if (debug)
    //     std::cout << "\n" << "Reading block: " << block_name << "\n";
}

static inline void read_layer_header(std::ifstream& fin, bool debug = false) {
    std::string layer_name;
    int32_t layer_name_size;
    fin.read(reinterpret_cast<char*>(&layer_name_size), sizeof(layer_name_size));
    layer_name.resize(layer_name_size);
    fin.read(reinterpret_cast<char*>(layer_name.data()), layer_name_size);

    // if (debug)
    //     std::cout << "Layer: " << layer_name << "\n";
}

static inline gten::Tensor read_weight(
    std::ifstream& fin, std::initializer_list<int> shape, bool debug = false)
{
    std::string weight_name;
    int32_t weight_name_size;
    fin.read(reinterpret_cast<char*>(&weight_name_size), sizeof(weight_name_size));
    weight_name.resize(weight_name_size);
    fin.read(reinterpret_cast<char*>(weight_name.data()), weight_name_size);

    int32_t weight_payload_size;
    fin.read(reinterpret_cast<char*>(&weight_payload_size), sizeof(weight_payload_size));

    // if (debug)
    //     std::cout << weight_name << " (" << weight_payload_size << ")\n";

    gten::Dtype dtype = gten::kFloat16;

    gten::Tensor tensor{shape, dtype};
    GTEN_ASSERT(
        static_cast<size_t>(weight_payload_size) == tensor.nbytes(),
        "Weight `%s` data size: %ld does not match the expected size: %d.",
        weight_name.c_str(), tensor.nbytes(), weight_payload_size);
    fin.read(tensor.data_ptr<char>(), weight_payload_size);

    return tensor;
}

static inline void read_into_weight(
    std::ifstream& fin, gten::Tensor& tensor, bool debug = false)
{
    std::string weight_name;
    int32_t weight_name_size;
    fin.read(reinterpret_cast<char*>(&weight_name_size), sizeof(weight_name_size));
    weight_name.resize(weight_name_size);
    fin.read(reinterpret_cast<char*>(weight_name.data()), weight_name_size);

    int32_t weight_payload_size;
    fin.read(reinterpret_cast<char*>(&weight_payload_size), sizeof(weight_payload_size));

    // if (debug)
        // std::cout << weight_name << " (" << weight_payload_size << ")\n";

    GTEN_ASSERT(
        static_cast<size_t>(weight_payload_size) == tensor.nbytes(),
        "Weight `%s` data size: %ld does not match the expected size: %d.",
        weight_name.c_str(), tensor.nbytes(), weight_payload_size);
    fin.read(tensor.data_ptr<char>(), weight_payload_size);
}

GPT2::GPT2(const GPT2Config& config, int max_ctx, GPT2Tokenizer&& tokenizer)
    : config_{config},
      wte_{Embedding(config.n_vocab, config.n_embed, max_ctx)},
      wpe_{PosEmbedding(config.n_ctx, config.n_embed)},
      ln_f_{LayerNorm(max_ctx, config.n_embed)},
      res_{Residual(max_ctx, config.n_embed)}
{
    tokenizer_ = std::move(tokenizer);
    blocks_.reserve(config.n_layer);
    for (int i = 0; i < config.n_layer; i++) {
        blocks_.push_back(ResidualAttnBlock(config_.n_head, config_.n_embed, 4*config_.n_embed, max_ctx, /*mask_attn=*/true));
    }
}


GPT2* load_gpt(const InferenceOptions &inference_opts, bool show_load_info)
{
    std::ifstream fin(inference_opts.get_model_path(), std::ios::binary);
    GTEN_ASSERT(fin.is_open(), "Failed to open model file: %s\n", inference_opts.get_model_path().c_str());
    
    int64_t load_time_ms = 0;
    gten::Timer timer(&load_time_ms);

    // Magic number.
    int64_t magic;
    fin.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    GTEN_ASSERT(magic == 0x454c49464e455447, "Magic number in the binary does not match the expected one.\n");
    
    // Model config.
    GPT2Config config;
    fin.read(reinterpret_cast<char*>(&config), sizeof(config));
    std::cout << config;

    // Vocab
    std::string vocab_segment_name;
    int32_t vocab_segment_name_size;
    int32_t vocab_segment_size;
    fin.read(reinterpret_cast<char*>(&vocab_segment_name_size), sizeof(vocab_segment_name_size));
    vocab_segment_name.resize(vocab_segment_name_size);
    fin.read(reinterpret_cast<char*>(vocab_segment_name.data()), vocab_segment_name_size);
    fin.read(reinterpret_cast<char*>(&vocab_segment_size), sizeof(vocab_segment_size));
    if (inference_opts.debug_mode)
        std::cout << "Read segment: [" << vocab_segment_name << "](" << vocab_segment_size << " bytes)\n";
    
    // Tokenizer.
    GPT2Tokenizer tokenizer{fin};
    const int num_prompt_tokens = tokenizer.encode(inference_opts.prompt).size();
    const int max_ctx = inference_opts.calculate_max_ctx_size(num_prompt_tokens);

    GPT2* model = new GPT2{config, max_ctx, std::move(tokenizer)};

    // WTE
    read_layer_header(fin, inference_opts.debug_mode);
    read_into_weight(fin, model->wte_.weight);

    // WPE
    read_layer_header(fin, inference_opts.debug_mode);
    read_into_weight(fin, model->wpe_.weight);


    // BLOCKS
    for (auto& block : model->blocks_)
    {
        read_block_header(fin, inference_opts.debug_mode);

        // Query projection layer.
        read_layer_header(fin, inference_opts.debug_mode);
        read_into_weight(fin, block.attn.query.weight);
        read_into_weight(fin, block.attn.query.bias);

        // Key projection layer.
        read_layer_header(fin, inference_opts.debug_mode);
        read_into_weight(fin, block.attn.key.weight);
        read_into_weight(fin, block.attn.key.bias);

        // Value projection layer.
        read_layer_header(fin, inference_opts.debug_mode);
        read_into_weight(fin, block.attn.value.weight);
        read_into_weight(fin, block.attn.value.bias);

        // QKV_out projection layer.
        read_layer_header(fin, inference_opts.debug_mode);
        read_into_weight(fin, block.attn.qkv_proj.weight);
        read_into_weight(fin, block.attn.qkv_proj.bias);

        // Input layernorm.
        read_layer_header(fin, inference_opts.debug_mode);
        read_into_weight(fin, block.attn_ln.weight);
        read_into_weight(fin, block.attn_ln.bias);

        // MLP fully-connected layer.
        read_layer_header(fin, inference_opts.debug_mode);
        read_into_weight(fin, block.mlp_fc.weight);
        read_into_weight(fin, block.mlp_fc.bias);

        // MLP out projection layer.
        read_layer_header(fin, inference_opts.debug_mode);
        read_into_weight(fin, block.mlp_proj.weight);
        read_into_weight(fin, block.mlp_proj.bias);

        // Attention layernorm.
        read_layer_header(fin, inference_opts.debug_mode);
        read_into_weight(fin, block.mlp_ln.weight);
        read_into_weight(fin, block.mlp_ln.bias);
    }
    
    // Block output Layernorm.
    read_layer_header(fin, inference_opts.debug_mode);
    read_into_weight(fin, model->ln_f_.weight);
    read_into_weight(fin, model->ln_f_.bias);

    timer.stop();
    model->time_load_ms_ = load_time_ms;

    return model;
}
