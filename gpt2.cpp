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
-f32:      Use 32-bit floating point values for inference instead of f16 which is the default.
-debug   : See debug-level information.
--temp T : Temperature to use during sampling. It must be greater than 0. [default=0.9].
--len  L : Number of words to generate. Minimum is 1 and max is 1000. [default=250].

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
    std::string prompt = "";
    std::string model_name = "Gpt2-medium";
    std::string inference_mode = "f16";
    double temperature = 0.9;
    int gen_tokens = 1000;
    bool debug = false;
    for (int i = 1; i < argc; i++)
    {
        std::string_view arg(argv[i]);
        if (arg == "-p") {
            if (argc <= i+1) {
                std::cout << "Prompt is missing.\n";
                return -1;
            }
            prompt = argv[i+1];
            i += 1;
        }
        else if (arg == "-sm") {
            model_name = "Gpt2";
        }
        else if (arg == "-lg") {
            model_name = "Gpt2-large";
        }
        else if (arg == "-f32") {
            inference_mode = "f32";
        }
        else if (arg == "-q8") {
            inference_mode = "q8";
        }
        else if (arg == "-debug") {
            debug = true;
        }
        else if (arg == "--temp") {
            if (argc <= i+1) {
                std::cout << "Temp value is missing.\n";
                return -1;
            }
            double temp;
            try {
                temp = std::stod(argv[i+1]);
            } catch (...) {
                std::cout << "Invalid temp value.\n";
                return -1;
            }
            if (temp <= 0.0f) {
                std::cout << "Temp value must be greater than zero.\n";
                return -1;
            }
            temperature = temp;
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
            if (len < 1 || len > 1000) {
                std::cout << "Length must be greater than 1 and less than 1000.\n";
                return -1;
            }
            gen_tokens = len;
            i += 1;
        }
        else {
            std::cout << "Unknown option: " << arg << "\n";
            return -1;
        }
    }
    if (prompt == "") {
        std::cout << "Prompt is not provided.\n";
		std::cout << usage << "\n";
		return -1;
    }

#if defined(__WIN32__) || defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
    std::string command = std::string("python model_dl.py ") + model_name + " " + inference_mode;
#else
    std::string command = std::string("python3 model_dl.py ") + model_name + " " + inference_mode;
#endif
    int res = std::system(command.c_str());
    if (res != 0) {
        std::cout << "Error: Failed to download " << model_name << " model due to network issues.\n";
        return -1;
    }

    std::string model_dir = std::string("models/") + model_name + ".fp16.gten";;
    if (inference_mode == "f32")
        model_dir = std::string("models/") + model_name + ".fp32.gten";
    else if (inference_mode == "q8")
        model_dir = std::string("models/") + model_name + ".q8.gten";

    if (debug) {
        std::cout << "Model name     : " << model_name << "\n";
        std::cout << "Model path     : " << model_dir << "\n";
        std::cout << "Inference      : " << inference_mode << "\n";
        std::cout << "Temperature    : " << temperature << "\n";
        std::cout << "Gen toks len   : " << gen_tokens << "\n";
    }

    GPT2 model(model_dir, debug);

    model.sample(prompt, temperature, gen_tokens);

    return 0;
}


GPT2::GPT2(const std::string &fname, bool show_load_info)
{
    load_from_file(fname, show_load_info);
}

gten::Tensor GPT2::logits(const gten::Tensor &inp)
{
    gten::Tensor logits;

    logits = res_.forward(wte_.forward(inp), wpe_.forward(inp.size(0)));
    for (auto &block : blocks_)
        logits = block.forward(logits);
    logits = ln_f_.forward(logits);
    logits = wte_.forward_proj(logits);

    return logits;
}

// Used for debugging purposes.
void GPT2::greedy_sample(const std::string &prompt, double temp, int max_iter)
{
    time_sample_ms_ = 0;


    std::vector<int32_t> tokens = tokenizer.encode(prompt);
    tokens.reserve(max_iter);
    gten::Tensor logits;
    const int logits_size = 50257;

    const int eot_token = 50256;
    const int initial_pos = tokens.size() - 1;
    const int n_iter = max_iter - tokens.size();
    int64_t niter = 0;
    // Use cerr because it is unbuffered.
    std::cerr << "\n\n" << prompt;
    for (int i = initial_pos; i < n_iter; i++)
    {
        // TODO: allow creation of tensors with external non-owning data.
        gten::Tensor input(tokens.data(), {(int32_t)tokens.size()}, gten::kInt32);
        logits = this->logits(input);

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

void GPT2::sample(const std::string& prompt, double temp, int ntokens)
{
    time_sample_ms_ = 0;

    const int max_ctx_size = 1000;
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<int32_t> tokens = tokenizer.encode(prompt);
    if (tokens.size() >= max_ctx_size) {
        // Prompt length is too large, quit. Technically, we can allow generation of
        // arbitrary-length documents by selecting the last 1000 context tokens and using
        // that to predict the next token but the modules are not yet designed with that
        // in mind. In the future that feature will be available.
        GTEN_ASSERT(false, "Prompt length is too large!");
    }
    tokens.reserve(ntokens + tokens.size());
    const int logits_size = 50257;
    std::vector<std::pair<double, int>> logits_probs;
    logits_probs.reserve(logits_size);
    const int eot_token = 50256;
    const int initial_pos = tokens.size() - 1;
    // If the requested ntokens + ctx_size > max_prompt_size, generate up to
    // max_prompt_size else generate up to requested size.
    const int max_iter = (ntokens + tokens.size()) >= max_ctx_size ? max_ctx_size : ntokens + tokens.size();
	int64_t niter = 0;
    // Use cerr because it is unbuffered.
    std::cerr << "\n\n" << prompt;
    for (int i = initial_pos; i < max_iter; i++)
    {
        // TODO: allow creation of tensors with external non-owning data.
        gten::Tensor input(tokens.data(), {(int32_t)tokens.size()}, gten::kInt32);
        gten::Tensor logits = this->logits(input);

        gten::Timer timer(&time_sample_ms_);
        const float *logits_data = logits.data_ptr<float>();

        logits_probs.clear();
        for (int j = 0; j < logits_size; ++j)
            logits_probs.push_back(std::make_pair((double)logits_data[j] / temp, j));

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
    int64_t total = emb_time + emb_proj_time + wpe_time + linear_time + attn_time + ln_time + gelu_time + res_time + time_sample_ms_;

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
    std::cout << "TOTAL MEMORY   : " << gten::Tensor::total_memory_allocated / 1000000 << "MB\n";
    std::cout << "--------------------------------------\n";
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
 *     <activation_dtype_size>
 *     <activation_scale: 4bytes, zerop: 4bytes> // Weight quantization params. Ignore if
 *     <weight_name_size>                        // activation_dtype_size != 1.
 *     <weight_name>
 *     <weight_dtype_size>
 *     <weight_scale: 4bytes, weight_zerop: 4bytes> // Weight quantization params. Ignore
 *     <weight_nbytes>                              // if weight_dtype_size != 1.
 *     <weight_bytes>
 * 
 *     if layer.has_bias:
 *         <b_name_size>
 *         <b_name>
 *         <b_dtype_size>
 *         <b_scale: 4bytes, wi_zerop: 4bytes>
 *         <b_nbytes>
 *         <b_bytes>
*/

static inline void read_block_header(std::ifstream& fin, bool debug = false)
{
    std::string block_name;
    int32_t block_name_size;
    fin.read(reinterpret_cast<char*>(&block_name_size), sizeof(block_name_size));
    block_name.resize(block_name_size);
    fin.read(reinterpret_cast<char*>(block_name.data()), block_name_size);

    if (debug)
        std::cout << "\n" << "Reading block: " << block_name << "\n";
}

static inline gten::AcvConfig read_layer_header(std::ifstream& fin, bool debug = false) {
    std::string layer_name;
    int32_t layer_name_size;
    fin.read(reinterpret_cast<char*>(&layer_name_size), sizeof(layer_name_size));
    layer_name.resize(layer_name_size);
    fin.read(reinterpret_cast<char*>(layer_name.data()), layer_name_size);

    int32_t activ_dsize;
    float activ_scale;
    int32_t activ_zerop;
    fin.read(reinterpret_cast<char*>(&activ_dsize), sizeof(activ_dsize));
    fin.read(reinterpret_cast<char*>(&activ_scale), sizeof(activ_scale));
    fin.read(reinterpret_cast<char*>(&activ_zerop), sizeof(activ_zerop));

    if (debug)
        std::cout << "Layer: "   << layer_name
                  << " [activ_dsize="  << activ_dsize
                  << ", activ_scale=" << activ_scale
                  << ", activ_zerop=" << activ_zerop
                  << "]\n";

    gten::Dtype dtype;
    if (activ_dsize == 1)
        dtype = gten::kQint8;
    else if (activ_dsize == 2)
        dtype = gten::kFloat16;
    else if (activ_dsize == 4)
        dtype = gten::kFloat32;
    else
        GTEN_ASSERT(false, "Unknown weight dsize: %d\n", activ_dsize);
    return gten::AcvConfig{dtype, activ_scale, activ_zerop};
}

static inline gten::Tensor read_weight(std::ifstream& fin, const std::vector<int>& shape, bool debug = false) {
    std::string weight_name;
    int32_t weight_name_size;
    fin.read(reinterpret_cast<char*>(&weight_name_size), sizeof(weight_name_size));
    weight_name.resize(weight_name_size);
    fin.read(reinterpret_cast<char*>(weight_name.data()), weight_name_size);

    int32_t weight_dsize;
    float weight_scale;
    int32_t weight_zerop;
    int32_t weight_payload_size;
    fin.read(reinterpret_cast<char*>(&weight_dsize), sizeof(weight_dsize));
    fin.read(reinterpret_cast<char*>(&weight_scale), sizeof(weight_scale));
    fin.read(reinterpret_cast<char*>(&weight_zerop), sizeof(weight_zerop));
    fin.read(reinterpret_cast<char*>(&weight_payload_size), sizeof(weight_payload_size));

    if (debug)
        std::cout << "     ~ "   << weight_name
                  << " (" << weight_payload_size
                  << ")[dsize="  << weight_dsize
                  << ", scale=" << weight_scale
                  << ", zerop=" << weight_zerop
                  << "]\n";

    gten::Dtype dtype;
    if (weight_dsize == 1)
        dtype = gten::kQint8;
    else if (weight_dsize == 2)
        dtype = gten::kFloat16;
    else if (weight_dsize == 4)
        dtype = gten::kFloat32;
    else
        GTEN_ASSERT(false, "Unknown weight dsize: %d\n", weight_dsize);

    gten::Tensor tensor{shape, dtype, weight_scale, weight_zerop};
    GTEN_ASSERT(
        weight_payload_size == tensor.nbytes(),
        "Weight `%s` data size: %ld does not match the expected size: %d.",
        weight_name.c_str(), tensor.nbytes(), weight_payload_size);
    fin.read(tensor.data_ptr<char>(), weight_payload_size);

    return tensor;
}

void GPT2::load_from_file(const std::string &fname, bool show_load_info)
{
	using namespace gten;

    std::ifstream fin(fname, std::ios::binary);
    GTEN_ASSERT(fin.is_open(), "Failed to open model file: %s\n", fname.c_str());

    if (show_load_info)
        std::cout << "Loading from   : " << fname.c_str() << "\n";
    gten::Timer timer(&time_load_ms_);

    // Magic number.
    int64_t magic;
    fin.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    GTEN_ASSERT(magic == magic_number_, "Magic number in file %s does not match the expected one.\n", fname.c_str());
    
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
    if (show_load_info)
        std::cout << "Read segment: [" << vocab_segment_name << "](" << vocab_segment_size << " bytes)\n";

    // Tokenizer.
    tokenizer = std::move(gten::GPT2Tokenizer(fin));

    // WTE
    AcvConfig wte_acv_config = read_layer_header(fin, show_load_info);
    Tensor wte_weight = read_weight(fin, {config_.n_vocab, config_.n_embed}, show_load_info);
    wte_ = gten::Embedding{wte_weight, wte_acv_config, config_.n_ctx};

    // WPE
    AcvConfig wpe_acv_config = read_layer_header(fin, show_load_info);
    Tensor wpe_weight = read_weight(fin, {config_.n_ctx, config_.n_embed}, show_load_info);
    wpe_ = PosEmbedding{wpe_weight, wpe_acv_config, config_.n_ctx};

    // (WTE + WPE) residual layer.
    AcvConfig res_acv_config = read_layer_header(fin, show_load_info);
    res_ = Residual{res_acv_config, config_.n_ctx, config_.n_embed};

    // BLOCKS
    blocks_ = std::vector<ResidualAttentionBlock>();
    blocks_.reserve(config_.n_layer);
    for (int64_t i = 0; i < config_.n_layer; i++)
    {
        read_block_header(fin, show_load_info);

        // Query projection layer.
        const AcvConfig q_acv_config = read_layer_header(fin, show_load_info);
        Tensor qw = read_weight(fin, {config_.n_embed, config_.n_embed}, show_load_info);
        Tensor qb = read_weight(fin, {config_.n_embed}, show_load_info);
        Linear query{qw, qb, q_acv_config, config_.n_ctx};

        // Key projection layer.
        const AcvConfig k_acv_config = read_layer_header(fin, show_load_info);
        Tensor kw = read_weight(fin, {config_.n_embed, config_.n_embed}, show_load_info);
        Tensor kb = read_weight(fin, {config_.n_embed}, show_load_info);
        Linear key{kw, kb, k_acv_config, config_.n_ctx};

        // Value projection layer.
        const AcvConfig v_acv_config = read_layer_header(fin, show_load_info);
        Tensor vw = read_weight(fin, {config_.n_embed, config_.n_embed}, show_load_info);
        Tensor vb = read_weight(fin, {config_.n_embed}, show_load_info);
        Linear value{vw, vb, v_acv_config, config_.n_ctx};

        // QKV_out projection layer.
        const AcvConfig attn_proj_acv_config = read_layer_header(fin, show_load_info);
        Tensor attn_projw = read_weight(fin, {config_.n_embed, config_.n_embed}, show_load_info);
        Tensor attn_projb = read_weight(fin, {config_.n_embed}, show_load_info);
        Linear out_proj{attn_projw, attn_projb, attn_proj_acv_config, config_.n_ctx};

        // Input layernorm.
        const AcvConfig ln_1_acv_config = read_layer_header(fin, show_load_info);
        Tensor ln_1w = read_weight(fin, {config_.n_embed}, show_load_info);
        Tensor ln_1b = read_weight(fin, {config_.n_embed}, show_load_info);
        LayerNorm ln_1{ln_1w, ln_1b, ln_1_acv_config, config_.n_ctx};

        // MLP fully-connected layer.
        const AcvConfig mlp_fc_acv_config = read_layer_header(fin, show_load_info);
        Tensor mlp_fcw = read_weight(fin, {4 * config_.n_embed, config_.n_embed}, show_load_info);
        Tensor mlp_fcb = read_weight(fin, {4 * config_.n_embed}, show_load_info);
        Linear mlp_fc{mlp_fcw, mlp_fcb, mlp_fc_acv_config, config_.n_ctx};

        // MLP out projection layer.
        const AcvConfig mlp_proj_acv_config = read_layer_header(fin, show_load_info);
        Tensor mlp_projw = read_weight(fin, {config_.n_embed, 4 * config_.n_embed}, show_load_info);
        Tensor mlp_projb = read_weight(fin, {config_.n_embed}, show_load_info);
        Linear mlp_proj{mlp_projw, mlp_projb, mlp_proj_acv_config, config_.n_ctx};

        // Attention layernorm.
        const AcvConfig ln_2_acv_config = read_layer_header(fin, show_load_info);
        Tensor ln_2w = read_weight(fin, {config_.n_embed}, show_load_info);
        Tensor ln_2b = read_weight(fin, {config_.n_embed}, show_load_info);
        LayerNorm ln_2{ln_2w, ln_2b, ln_2_acv_config, config_.n_ctx};

        // Multihead self attn activ & GELU activ.
        AcvConfig self_attn_acv_config = read_layer_header(fin, show_load_info);
        MultiHeadSelfAttn self_attn{query, key, value, out_proj, self_attn_acv_config, config_.n_ctx, config_.n_embed, config_.n_head};

        // GELU.
        AcvConfig gelu_acv_config = read_layer_header(fin, show_load_info);
        GELU gelu{gelu_acv_config, config_.n_ctx, config_.n_embed * 4};

        // Attn inp residual.
        AcvConfig inp_res_acv_config = read_layer_header(fin, show_load_info);
        Residual inp_res{inp_res_acv_config, config_.n_ctx, config_.n_embed};
        AcvConfig attn_res_acv_config = read_layer_header(fin, show_load_info);
        Residual attn_res{attn_res_acv_config, config_.n_ctx, config_.n_embed};

        ResidualAttentionBlock bl(self_attn, ln_1, mlp_fc, mlp_proj, ln_2, gelu, inp_res, attn_res, config_.n_ctx, config_.n_embed);
        blocks_.push_back(std::move(bl));
    }
    
    // Block output Layernorm.
    const AcvConfig ln_f_acv_config = read_layer_header(fin, show_load_info);
    Tensor ln_fw = read_weight(fin, {config_.n_embed}, show_load_info);
    Tensor ln_fb = read_weight(fin, {config_.n_embed}, show_load_info);
    ln_f_ = LayerNorm{ln_fw, ln_fb, ln_f_acv_config, config_.n_ctx};
}
