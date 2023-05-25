#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <random>

#include "gpt2.h"



const char *usage = R"(
usage: gpt2 [options] -p PROMPT

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

    if (options.prompt == "") {
        std::cout << "Prompt is not provided.\n";
		std::cout << usage << "\n";
		return -1;
    }
    options.print_debug_info();

    int res = std::system(options.get_dl_command().c_str());
    if (res != 0) {
        std::cout << "Error: Failed to download " << options.model_name << " model due to network issues.\n";
        return -1;
    }

    GPT2 model{options, false};

    if (options.greedy)
        model.greedy_sample(options);
    else
        model.sample(options);

    return 0;
}


GPT2::GPT2(const InferenceOptions& inference_opts, bool show_load_info)
{
    load_from_file(inference_opts, show_load_info);
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

// Used for debugging purposes.
void GPT2::greedy_sample(const InferenceOptions& opts)
{
    time_sample_ms_ = 0;

    const int max_ctx_size = 128;

    std::vector<int32_t> tokens = tokenizer.encode(opts.prompt);
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
        std::cerr << tokenizer.decode(maxi);
        tokens.push_back(maxi);

        niter += 1;
    }
    std::cerr << "\n";

    show_performance(niter);
}

void GPT2::sample(const InferenceOptions& opts)
{
    time_sample_ms_ = 0;

    const int max_ctx_size = 1024;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<int32_t> tokens = tokenizer.encode(opts.prompt);
    tokens.reserve(max_ctx_size);
    const int logits_size = 50257;
    std::vector<std::pair<double, int>> logits_probs;
    logits_probs.reserve(logits_size);
    const int eot_token = 50256;
    const int initial_pos = tokens.size() - 1;

    // Total ntokens = Requested number of tokens + prompt num of tokens.
    int total_ntokens = opts.gen_tokens + tokens.size();
    // If the total_ntokens > max_prompt_size, generate up to
    // max_prompt_size. Else generate up to requested size.
    const int max_iter = total_ntokens >= max_ctx_size ? max_ctx_size : total_ntokens;
	int64_t niter = 0;
    // Use cerr because it is unbuffered.
    std::cerr << "\n\n" << opts.prompt;
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
        std::cerr << tokenizer.decode(maxi);
        tokens.push_back(maxi);

        niter += 1;
    }
    std::cerr << "\n";

    if (opts.debug_mode)
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

static int64_t calculate_mem_fp16(const GPT2Config& cfg, int max_ctx) {
    int64_t weights_mem = (cfg.n_vocab * cfg.n_embed)
      + (cfg.n_ctx * cfg.n_embed)
      + cfg.n_layer * (12 * cfg.n_embed * cfg.n_embed + 13 * cfg.n_embed)
      + (2 * cfg.n_embed);

    int64_t acv_mem = cfg.n_vocab * 2
      + (3 * max_ctx * cfg.n_embed)
      + (cfg.n_layer * (10 * max_ctx * cfg.n_embed + cfg.n_head * max_ctx * max_ctx + 2 * max_ctx * 4 * cfg.n_embed));
    
    int64_t total_mem = 2 * (weights_mem + acv_mem);

    return total_mem;
}

/** Gpt2 Model Binary format.
 * 
 * Key:
 *    <LABEL> = LABEL occupies 4 bytes in the binary.
 *    {LABEL: SIZE} = LABEL occupies SIZE bytes.
 * 
 * FORMAT:
 * {GTEN_magic_number: 8} = GTENFILE in ascii.
 * <n_vocab>
 * <max_ctx>
 * <n_embed>
 * <n_layer>
 * <n_head>
 * <size_vocab_idname>
 * {vocab_idname: size_vocab_idname}
 * <size_vocab>
 * {vocab: size_vocab}
 *
 * for layer in (wte, wpe, block_0, ... block_N, ln_f)
 *     [block_name_size]. Not included for wte, wpe and ln_f.
 *     [block_name]. Not included for wte, wpe and ln_f.
 *     <layer_name_size>
 *     <layer_name>
 *     <weight_name_size>
 *     <weight_name>
 *     <weight_nbytes>
 *     <weight_bytes>
 * 
 *     if layer.has_bias:
 *         <b_name_size>
 *         <b_name>
 *         <b_nbytes>
 *         <b_bytes>
 *
*/

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
    std::ifstream& fin, gten::TensorMemoryPool& pool, std::initializer_list<int> shape, bool debug = false) {
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

    gten::Tensor tensor{pool, shape, dtype};
    GTEN_ASSERT(
        weight_payload_size == tensor.nbytes(),
        "Weight `%s` data size: %ld does not match the expected size: %d.",
        weight_name.c_str(), tensor.nbytes(), weight_payload_size);
    fin.read(tensor.data_ptr<char>(), weight_payload_size);

    return tensor;
}

void GPT2::load_from_file(const InferenceOptions& inference_opts, bool show_load_info)
{
	using namespace gten;

    std::ifstream fin(inference_opts.get_model_path(), std::ios::binary);
    GTEN_ASSERT(fin.is_open(), "Failed to open model file: %s\n", inference_opts.get_model_path().c_str());
    
    gten::Timer timer(&time_load_ms_);

    // Magic number.
    int64_t magic;
    fin.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    GTEN_ASSERT(magic == magic_number_, "Magic number in the binary does not match the expected one.\n");
    
    // Model config.
    config_ = GPT2Config();
    fin.read(reinterpret_cast<char*>(&config_), sizeof(config_));
    std::cout << config_;

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
    tokenizer = std::move(gten::GPT2Tokenizer(fin));

    const int num_prompt_tokens = tokenizer.encode(inference_opts.prompt).size();
    const int max_ctx = inference_opts.calculate_max_ctx_size(num_prompt_tokens);
    int64_t req_mem = calculate_mem_fp16(config_, max_ctx);
    TensorMemoryPool pool{req_mem};

    // WTE
    read_layer_header(fin, inference_opts.debug_mode);
    Tensor wte_weight = read_weight(fin, pool, {config_.n_vocab, config_.n_embed}, inference_opts.debug_mode);
    const Tensor emb_acv{pool, {max_ctx, config_.n_embed}, kFloat16};
    const Tensor emb_proj_acv{pool, {config_.n_vocab}, kFloat32};
    wte_ = gten::Embedding{wte_weight, emb_acv, emb_proj_acv};

    // WPE
    read_layer_header(fin, inference_opts.debug_mode);
    Tensor wpe_weight = read_weight(fin, pool, {config_.n_ctx, config_.n_embed}, inference_opts.debug_mode);
    wpe_ = PosEmbedding{wpe_weight, config_.n_ctx};

    // (WTE + WPE) residual layer.
    const Tensor res_acv{pool, {max_ctx, config_.n_embed}, kFloat16};
    res_ = Residual{res_acv};

    // BLOCKS
    blocks_ = std::vector<ResidualAttentionBlock>();
    blocks_.reserve(config_.n_layer);
    for (int64_t i = 0; i < config_.n_layer; i++)
    {
        read_block_header(fin, inference_opts.debug_mode);

        // Query projection layer.
        read_layer_header(fin, inference_opts.debug_mode);
        Tensor qw = read_weight(fin, pool, {config_.n_embed, config_.n_embed}, inference_opts.debug_mode);
        Tensor qb = read_weight(fin, pool, {config_.n_embed}, inference_opts.debug_mode);
        const Tensor q_acv{pool, {max_ctx, config_.n_embed}, kFloat16};
        Linear query{qw, qb, q_acv};

        // Key projection layer.
        read_layer_header(fin, inference_opts.debug_mode);
        Tensor kw = read_weight(fin, pool, {config_.n_embed, config_.n_embed}, inference_opts.debug_mode);
        Tensor kb = read_weight(fin, pool, {config_.n_embed}, inference_opts.debug_mode);
        const Tensor k_acv{pool, {max_ctx, config_.n_embed}, kFloat16};
        Linear key{kw, kb, k_acv};

        // Value projection layer.
        read_layer_header(fin, inference_opts.debug_mode);
        Tensor vw = read_weight(fin, pool, {config_.n_embed, config_.n_embed}, inference_opts.debug_mode);
        Tensor vb = read_weight(fin, pool, {config_.n_embed}, inference_opts.debug_mode);
        const Tensor v_acv{pool, {max_ctx, config_.n_embed}, kFloat16};
        Linear value{vw, vb, v_acv};

        // QKV_out projection layer.
        read_layer_header(fin, inference_opts.debug_mode);
        Tensor attn_projw = read_weight(fin, pool, {config_.n_embed, config_.n_embed}, inference_opts.debug_mode);
        Tensor attn_projb = read_weight(fin, pool, {config_.n_embed}, inference_opts.debug_mode);
        const Tensor out_proj_acv{pool, {max_ctx, config_.n_embed}, kFloat16};
        Linear out_proj{attn_projw, attn_projb, out_proj_acv};

        // Input layernorm.
        read_layer_header(fin, inference_opts.debug_mode);
        Tensor ln_1w = read_weight(fin, pool, {config_.n_embed}, inference_opts.debug_mode);
        Tensor ln_1b = read_weight(fin, pool, {config_.n_embed}, inference_opts.debug_mode);
        const Tensor ln_1_acv{pool, {max_ctx, config_.n_embed}, kFloat16};
        LayerNorm ln_1{ln_1w, ln_1b, ln_1_acv};

        // MLP fully-connected layer.
        read_layer_header(fin, inference_opts.debug_mode);
        Tensor mlp_fcw = read_weight(fin, pool, {4 * config_.n_embed, config_.n_embed}, inference_opts.debug_mode);
        Tensor mlp_fcb = read_weight(fin, pool, {4 * config_.n_embed}, inference_opts.debug_mode);
        const Tensor mlp_fc_acv{pool, {max_ctx, 4 * config_.n_embed}, kFloat16};
        Linear mlp_fc{mlp_fcw, mlp_fcb, mlp_fc_acv};

        // MLP out projection layer.
        read_layer_header(fin, inference_opts.debug_mode);
        Tensor mlp_projw = read_weight(fin, pool, {config_.n_embed, 4 * config_.n_embed}, inference_opts.debug_mode);
        Tensor mlp_projb = read_weight(fin, pool, {config_.n_embed}, inference_opts.debug_mode);
        const Tensor mlp_proj_acv{pool, {max_ctx, config_.n_embed}, kFloat16};
        Linear mlp_proj{mlp_projw, mlp_projb, mlp_proj_acv};

        // Attention layernorm.
        read_layer_header(fin, inference_opts.debug_mode);
        Tensor ln_2w = read_weight(fin, pool, {config_.n_embed}, inference_opts.debug_mode);
        Tensor ln_2b = read_weight(fin, pool, {config_.n_embed}, inference_opts.debug_mode);
        const Tensor ln_2_acv{pool, {max_ctx, config_.n_embed}, kFloat16};
        LayerNorm ln_2{ln_2w, ln_2b, ln_2_acv};

        // Multihead self attn activ.
        const Tensor qk_acv{pool, {config_.n_head, max_ctx, max_ctx}, kFloat16};
        const Tensor qkv_acv{pool, {max_ctx, config_.n_embed}, kFloat16};
        MultiHeadSelfAttn self_attn{query, key, value, out_proj, qk_acv, qkv_acv, config_.n_head};

        // GELU acv.
        const Tensor gelu_acv{pool, {max_ctx, 4 * config_.n_embed}, kFloat16};
        GELU gelu{gelu_acv};

        // Attn inp residual acv.
        const Tensor inp_res_acv{pool, {max_ctx, config_.n_embed}, kFloat16};
        Residual inp_res{inp_res_acv};
        const Tensor attn_res_acv{pool, {max_ctx, config_.n_embed}, kFloat16};
        Residual attn_res{attn_res_acv};

        ResidualAttentionBlock bl(self_attn, ln_1, mlp_fc, mlp_proj, ln_2, gelu, inp_res, attn_res, config_.n_ctx, config_.n_embed);
        blocks_.push_back(std::move(bl));
    }
    
    // Block output Layernorm.
    read_layer_header(fin, inference_opts.debug_mode);
    Tensor ln_fw = read_weight(fin, pool, {config_.n_embed}, inference_opts.debug_mode);
    Tensor ln_fb = read_weight(fin, pool, {config_.n_embed}, inference_opts.debug_mode);
    const Tensor ln_f_acv{pool, {max_ctx, config_.n_embed}, kFloat16};
    ln_f_ = LayerNorm{ln_fw, ln_fb, ln_f_acv};
}
