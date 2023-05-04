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
    int gen_tokens = 250;
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
    std::cout << "--------------------------------------\n\n";
}

/** File format.
 * The GPT2 gten file format is designed to be simple and minimal. We pack vocab,
 * config and layer weight data in a single binary file. The different sections have
 * names to allow for debugging.
 * 
 *  number |     section       | size in bytes
 *  ------------------------------------------
 *    1    |      magic        | 8
 *    2    |  dtype_byte_size  | 8
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

void GPT2::load_from_file(const std::string &fname, bool show_load_info)
{
	using namespace gten;

    std::ifstream fin(fname, std::ios::binary);
    GTEN_ASSERT(fin.is_open(), "Failed to open model file: %s\n", fname.c_str());

    if (show_load_info)
        std::cout << "Loading from   : " << fname.c_str() << "\n";
    gten::Timer timer(&time_load_ms_);

    int64_t magic;
    fin.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    GTEN_ASSERT(magic == magic_number_, "Magic number in file %s does not match the expected one.\n", fname.c_str());

    int64_t weights_dtype_size;
    fin.read(reinterpret_cast<char*>(&weights_dtype_size), sizeof(weights_dtype_size));
    if (show_load_info)
        std::cout << "FP mode size   : " << weights_dtype_size << " bytes\n";
    InferenceMode inference_mode;
    if (weights_dtype_size == 2) {
        inference_mode = kFloat16;
        std::cerr << "Inference mode : FP16\n";
    }
    else if (weights_dtype_size == 4) {
        std::cerr << "Inference mode : FP32\n";
        inference_mode = kFloat32;
    }
    else
        GTEN_ASSERT(false, "Unknown weight dtype size: %ld", weights_dtype_size);
    
    config_ = GPT2Config();
    // We could just read the data in the config directly but the config stores
    // the data in int32_t type while the data in the input file is in int64_t type.
    // Changing the config data types to int64_t causes a lot of narrowing
    // conversions in modules.cpp. The solution to this is to promote all usage
    // of int32 to int64.
    int64_t temp_config_value;
    fin.read(reinterpret_cast<char*>(&temp_config_value), sizeof(temp_config_value));
    config_.n_vocab = static_cast<int32_t>(temp_config_value);
    fin.read(reinterpret_cast<char*>(&temp_config_value), sizeof(temp_config_value));
    config_.n_ctx = static_cast<int32_t>(temp_config_value);
    fin.read(reinterpret_cast<char*>(&temp_config_value), sizeof(temp_config_value));
    config_.n_embed = static_cast<int32_t>(temp_config_value);
    fin.read(reinterpret_cast<char*>(&temp_config_value), sizeof(temp_config_value));
    config_.n_layer = static_cast<int32_t>(temp_config_value);
    fin.read(reinterpret_cast<char*>(&temp_config_value), sizeof(temp_config_value));
    config_.n_head = static_cast<int32_t>(temp_config_value);
    std::cout << config_;

    // Vocab
    std::string segment_name;
    int64_t segment_name_size;
    int64_t segment_size;
    fin.read(reinterpret_cast<char*>(&segment_name_size), sizeof(segment_name_size));
    segment_name.resize(segment_name_size);
    fin.read(reinterpret_cast<char*>(segment_name.data()), segment_name_size);
    fin.read(reinterpret_cast<char*>(&segment_size), sizeof(segment_size));
    if (show_load_info)
        std::cout << "Reading segment: [" << segment_name << "](" << segment_size << " bytes)\n";

    tokenizer = std::move(gten::GPT2Tokenizer(fin));

    // WTE
    fin.read(reinterpret_cast<char*>(&segment_name_size), sizeof(segment_name_size));
    segment_name.resize(segment_name_size);
    fin.read(reinterpret_cast<char*>(segment_name.data()), segment_name_size);
    fin.read(reinterpret_cast<char*>(&segment_size), sizeof(segment_size));
    Tensor wte_weight({config_.n_vocab, config_.n_embed}, inference_mode);
    fin.read(wte_weight.data_ptr<char>(), segment_size);
    wte_ = gten::Embedding(inference_mode, wte_weight, config_.n_ctx);
    if (show_load_info)
        std::cout << "Reading segment: [" << segment_name << "](" << segment_size << " bytes)\n";

    // WPE
    fin.read(reinterpret_cast<char*>(&segment_name_size), sizeof(segment_name_size));
    segment_name.resize(segment_name_size);
    fin.read(reinterpret_cast<char*>(segment_name.data()), segment_name_size);
    fin.read(reinterpret_cast<char*>(&segment_size), sizeof(segment_size));
    Tensor wpe_weight({config_.n_ctx, config_.n_embed}, inference_mode);
    fin.read(wpe_weight.data_ptr<char>(), segment_size);
    wpe_ = PosEmbedding(inference_mode, wpe_weight, config_.n_ctx);
    if (show_load_info)
        std::cout << "Reading segment: [" << segment_name << "](" << segment_size << " bytes)\n";

    // BLOCKS
    blocks_ = std::vector<ResidualAttentionBlock>();
    blocks_.reserve(config_.n_layer);
    for (int64_t i = 0; i < config_.n_layer; i++)
    {
        fin.read(reinterpret_cast<char*>(&segment_name_size), sizeof(segment_name_size));
        segment_name.resize(segment_name_size);
        fin.read(reinterpret_cast<char*>(segment_name.data()), segment_name_size);
        fin.read(reinterpret_cast<char*>(&segment_size), sizeof(segment_size));
        if (show_load_info)
            std::cout << "Reading segment: [" << segment_name << "](" << segment_size << " bytes)\n";

        Tensor qw({config_.n_embed, config_.n_embed}, inference_mode);
        Tensor qb({config_.n_embed}, inference_mode);
        Tensor kw({config_.n_embed, config_.n_embed}, inference_mode);
        Tensor kb({config_.n_embed}, inference_mode);
        Tensor vw({config_.n_embed, config_.n_embed}, inference_mode);
        Tensor vb({config_.n_embed}, inference_mode);
        Tensor attn_projw({config_.n_embed, config_.n_embed}, inference_mode);
        Tensor attn_projb({config_.n_embed}, inference_mode);
        Tensor ln_1w({config_.n_embed}, inference_mode);
        Tensor ln_1b({config_.n_embed}, inference_mode);
        Tensor mlp_fcw({4 * config_.n_embed, config_.n_embed}, inference_mode);
        Tensor mlp_fcb({4 * config_.n_embed}, inference_mode);
        Tensor mlp_projw({config_.n_embed, 4 * config_.n_embed}, inference_mode);
        Tensor mlp_projb({config_.n_embed}, inference_mode);
        Tensor ln_2w({config_.n_embed}, inference_mode);
        Tensor ln_2b({config_.n_embed}, inference_mode);           

        fin.read(qw.data_ptr<char>(),         qw.numel()         * qw.itemsize());
        fin.read(qb.data_ptr<char>(),         qb.numel()         * qb.itemsize());
        fin.read(kw.data_ptr<char>(),         kw.numel()         * kw.itemsize());
        fin.read(kb.data_ptr<char>(),         kb.numel()         * kb.itemsize());
        fin.read(vw.data_ptr<char>(),         vw.numel()         * vw.itemsize());
        fin.read(vb.data_ptr<char>(),         vb.numel()         * vb.itemsize());
        fin.read(attn_projw.data_ptr<char>(), attn_projw.numel() * attn_projw.itemsize());
        fin.read(attn_projb.data_ptr<char>(), attn_projb.numel() * attn_projb.itemsize());
        fin.read(ln_1w.data_ptr<char>(),      ln_1w.numel()      * ln_1w.itemsize());
        fin.read(ln_1b.data_ptr<char>(),      ln_1b.numel()      * ln_1b.itemsize());
        fin.read(mlp_fcw.data_ptr<char>(),    mlp_fcw.numel()    * mlp_fcw.itemsize());
        fin.read(mlp_fcb.data_ptr<char>(),    mlp_fcb.numel()    * mlp_fcb.itemsize());
        fin.read(mlp_projw.data_ptr<char>(),  mlp_projw.numel()  * mlp_projw.itemsize());
        fin.read(mlp_projb.data_ptr<char>(),  mlp_projb.numel()  * mlp_projb.itemsize());
        fin.read(ln_2w.data_ptr<char>(),      ln_2w.numel()      * ln_2w.itemsize());
        fin.read(ln_2b.data_ptr<char>(),      ln_2b.numel()      * ln_2b.itemsize());

        Linear query(inference_mode, qw, qb, config_.n_ctx);
        Linear key(inference_mode, kw, kb, config_.n_ctx);
        Linear value(inference_mode, vw, vb, config_.n_ctx);
        Linear out_proj(inference_mode, attn_projw, attn_projb, config_.n_ctx);

        MultiHeadSelfAttn self_attn(inference_mode, query, key, value, out_proj, config_.n_ctx, config_.n_embed, config_.n_head);
        LayerNorm ln_1(inference_mode, ln_1w, ln_1b, config_.n_ctx);
        Linear mlp_fc(inference_mode, mlp_fcw, mlp_fcb, config_.n_ctx);
        Linear mlp_proj(inference_mode, mlp_projw, mlp_projb, config_.n_ctx);
        LayerNorm ln_2(inference_mode, ln_2w, ln_2b, config_.n_ctx);
        GELU gelu(inference_mode, config_.n_ctx, config_.n_embed * 4);

        ResidualAttentionBlock bl(inference_mode, self_attn, ln_1, mlp_fc, mlp_proj, ln_2, gelu, config_.n_ctx, config_.n_embed);
        blocks_.push_back(std::move(bl));
    }
    
    // LN_F
    fin.read(reinterpret_cast<char*>(&segment_name_size), sizeof(segment_name_size));
    segment_name.resize(segment_name_size);
    fin.read(reinterpret_cast<char*>(segment_name.data()), segment_name_size);
    fin.read(reinterpret_cast<char*>(&segment_size), sizeof(segment_size));
    if (show_load_info)
        std::cout << "Reading segment: [" << segment_name << "](" << segment_size << " bytes)\n";

    Tensor ln_fw({config_.n_embed}, inference_mode);
    Tensor ln_fb({config_.n_embed}, inference_mode);
    fin.read(ln_fw.data_ptr<char>(), ln_fw.numel() * ln_fw.itemsize());
    fin.read(ln_fb.data_ptr<char>(), ln_fb.numel() * ln_fb.itemsize());

    ln_f_ = LayerNorm(inference_mode, ln_fw, ln_fb, config_.n_ctx);
    res_ = Residual(inference_mode, config_.n_ctx, config_.n_embed);
}
