#include "gpt2.h"

#include <iostream>
#include <string_view>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>



const char *usage = R"(
usage: gpt2 [options] PROMPT

Options and arguments.
-sm : use small model (117M) for inference.
-md : Use medium model (345M) for inference. This model is chosen by default.
-lg : Use large model (762M) for inference.

Examples:
  ./gpt2 "Once upon a time" 
  ./gpt2 -lg "Once upon a time"
)";


int main(int argc, char const *argv[])
{
    if (argc < 2) {
        std::cout << "Prompt is not provided.\n";
		std::cout << usage << "\n";
		return -1;
    }
    std::string model_name = "GPT-2-345M";
    std::string prompt = "";
    for (int i = 1; i < argc; i++)
    {
        std::string_view arg(argv[i]);
        if (arg == "-sm") model_name = "GPT-2-117M";
        else if (arg == "-lg") model_name = "GPT-2-762M";
        else prompt = arg;
    }
    if (prompt == "") {
        std::cout << "Prompt is not provided.\n";
		std::cout << usage << "\n";
		return -1;
    }

#if defined(__WIN32__) || defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
    std::string command = std::string("python model_dl.py ") + model_name;
#else
    std::string command = std::string("python3 model_dl.py ") + model_name;
#endif
    int res = std::system(command.c_str());
    if (res != 0) {
        std::cout << "Error: Failed to download model due to network issues.\n";
        return -1;
    }

    std::string model_dir = std::string("models/") + model_name + ".gten";
    GPT2 model(model_dir);

    model.sample(prompt, 0.9, 1000);

    return 0;
}



// #########################################################################
// #########################################################################


GPT2::GPT2(const std::string &fname)
{
    load_from_file(fname);
}

gten::Tensor GPT2::logits(const gten::Tensor &inp)
{
    gten::Tensor logits;

    logits = res_.forward(wte_.forward(inp), wpe_.forward(inp.shape()[0]));
    for (auto &block : blocks_)
        logits = block.forward(logits);
    logits = ln_f_.forward(logits);
    logits = wte_.forward_project(logits);

    return logits;
}

void GPT2::sample(const std::string &prompt, double temp, int max_iter)
{
    GTEN_CHECK(max_iter <= config_.n_ctx, "max_iter, %d, cannot exceed maximum context length, %ld.\n", max_iter, config_.n_ctx);

    time_sample_ms_ = 0;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<int32_t> tokens = tokenizer.encode(prompt);
    tokens.reserve(max_iter);
    gten::Tensor logits;
    const int logits_size = 50257;
    std::vector<std::pair<double, int>> logits_probs;
    logits_probs.reserve(logits_size);

    const int eot_token = 50256;
    const int initial_pos = tokens.size() - 1;
    const int n_iter = max_iter - tokens.size();
	int64_t niter = 0;
    // Use cerr because it is unbuffered.
    std::cerr << prompt;
    for (int i = initial_pos; i < n_iter; i++)
    {
        gten::Tensor input(tokens.data(), {(uint32_t)tokens.size()}, gten::Dtype::Int32);
        logits = this->logits(input);

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
    if (niter < 0)
        return;
    std::cout << "\n";
    std::cout << "------------------------------------\n";
    std::cout << "LAYER/OP    TIME PER TOKEN  TIME TOTAL\n";
    std::cout << "------------------------------------\n";
    std::cout << "Embedding      | " << std::setw(2) << wte_.emb_time()/niter      << "ms | " << wte_.emb_time()      << "ms\n";
    std::cout << "Embedding proj | " << std::setw(2) << wte_.emb_proj_time()/niter << "ms | " << wte_.emb_proj_time() << "ms\n";
    std::cout << "Pos embedding  | " << std::setw(2) << wpe_.time()/niter          << "ms | " << wpe_.time()          << "ms\n";

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
    std::cout << "Linear (total) | " << std::setw(2) << linear_time/niter << "ms | " << linear_time << "ms\n";
    // std::cout << "Linear (qkv)   | " << std::setw(2) << attn_lin/niter    << "ms | " << attn_lin    << "ms\n";
    // std::cout << "Linear (mlp)   | " << std::setw(2) << mlpp_time/niter   << "ms | " << mlpp_time   << "ms\n";
    std::cout << "Attention      | " << std::setw(2) << attn_time/niter   << "ms | " << attn_time   << "ms\n";
    std::cout << "Layer norm     | " << std::setw(2) << ln_time/niter     << "ms | " << ln_time     << "ms\n";
    std::cout << "GELU           | " << std::setw(2) << gelu_time/niter   << "ms | " << gelu_time   << "ms\n";
    std::cout << "Residual       | " << std::setw(2) << res_time/niter    << "ms | " << res_time   << "ms\n";
    std::cout << "Sampler        | " << std::setw(2) << time_sample_ms_/niter    << "ms | " << time_sample_ms_   << "ms\n";
    std::cout << "------------------------------------\n\n";
}

/** File format.
 * The GPT2 gten file format is designed to be simple and minimal. We pack vocab,
 * config and layer weight data in a single binary file. The different sections have
 * names to allow for debugging.
 * 
 *  number |     section       | size in bytes
 *  ------------------------------------------
 *    1    |      magic        | 8
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
void GPT2::load_from_file(const std::string &fname)
{
	using namespace gten;

    std::ifstream fin(fname, std::ios::binary);
    GTEN_CHECK(fin.is_open(), "Failed to open model file: %s\n", fname.c_str());

    std::cout << "Loading from " << fname.c_str() << "\n";
    int64_t load_time;
    gten::Timer timer(&load_time);

    int64_t magic;
    fin.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    GTEN_CHECK(magic == magic_number_, "Magic number in file %s does not match the expected one.\n", fname.c_str());

    config_ = GPT2Config();
    fin.read(reinterpret_cast<char*>(&config_), sizeof(config_));
    std::cout << config_;

    // Vocab
    std::string segment_name;
    int64_t segment_name_size;
    int64_t segment_size;
    fin.read(reinterpret_cast<char*>(&segment_name_size), sizeof(segment_name_size));
    segment_name.resize(segment_name_size);
    fin.read(reinterpret_cast<char*>(segment_name.data()), segment_name_size);
    fin.read(reinterpret_cast<char*>(&segment_size), sizeof(segment_size));
    // std::cout << "Reading segment: [" << segment_name << "](" << segment_size << " bytes)\n";

    tokenizer = std::move(gten::GPT2Tokenizer(fin));

    // WTE
    fin.read(reinterpret_cast<char*>(&segment_name_size), sizeof(segment_name_size));
    segment_name.resize(segment_name_size);
    fin.read(reinterpret_cast<char*>(segment_name.data()), segment_name_size);
    fin.read(reinterpret_cast<char*>(&segment_size), sizeof(segment_size));
    gten::Tensor wte_weight({(uint32_t)config_.n_vocab, (uint32_t)config_.n_embed}, Dtype::Float32);
    fin.read(reinterpret_cast<char*>(wte_weight.data_ptr<void>()), segment_size);
    wte_ = gten::Embedding(wte_weight, config_.n_ctx);
    // std::cout << "Reading segment: [" << segment_name << "](" << segment_size << " bytes)\n";

    // WPE
    fin.read(reinterpret_cast<char*>(&segment_name_size), sizeof(segment_name_size));
    segment_name.resize(segment_name_size);
    fin.read(reinterpret_cast<char*>(segment_name.data()), segment_name_size);
    fin.read(reinterpret_cast<char*>(&segment_size), sizeof(segment_size));
    gten::Tensor wpe_weight({(uint32_t)config_.n_ctx, (uint32_t)config_.n_embed}, Dtype::Float32);
    fin.read(reinterpret_cast<char*>(wpe_weight.data_ptr<void>()), segment_size);
    wpe_ = gten::PosEmbedding(wpe_weight, config_.n_ctx);
    // std::cout << "Reading segment: [" << segment_name << "](" << segment_size << " bytes)\n";

    // BLOCKS
    blocks_ = std::vector<gten::ResidualAttentionBlock>();
    blocks_.reserve(config_.n_layer);
    for (int64_t i = 0; i < config_.n_layer; i++)
    {
        fin.read(reinterpret_cast<char*>(&segment_name_size), sizeof(segment_name_size));
        segment_name.resize(segment_name_size);
        fin.read(reinterpret_cast<char*>(segment_name.data()), segment_name_size);
        fin.read(reinterpret_cast<char*>(&segment_size), sizeof(segment_size));
        // std::cout << "Reading segment: [" << segment_name << "](" << segment_size << " bytes)\n";

        Tensor qw({(uint32_t)config_.n_embed, (uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor qb({(uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor kw({(uint32_t)config_.n_embed, (uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor kb({(uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor vw({(uint32_t)config_.n_embed, (uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor vb({(uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor attn_projw({(uint32_t)config_.n_embed, (uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor attn_projb({(uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor ln_1w({(uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor ln_1b({(uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor mlp_fcw({4 * (uint32_t)config_.n_embed, (uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor mlp_fcb({4 * (uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor mlp_projw({(uint32_t)config_.n_embed, 4 * (uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor mlp_projb({(uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor ln_2w({(uint32_t)config_.n_embed}, Dtype::Float32);
        Tensor ln_2b({(uint32_t)config_.n_embed}, Dtype::Float32);           

        fin.read(reinterpret_cast<char*>(qw.data_ptr<void>()),         qw.numel()         * qw.bytes_per_item());
        fin.read(reinterpret_cast<char*>(qb.data_ptr<void>()),         qb.numel()         * qb.bytes_per_item());
        fin.read(reinterpret_cast<char*>(kw.data_ptr<void>()),         kw.numel()         * kw.bytes_per_item());
        fin.read(reinterpret_cast<char*>(kb.data_ptr<void>()),         kb.numel()         * kb.bytes_per_item());
        fin.read(reinterpret_cast<char*>(vw.data_ptr<void>()),         vw.numel()         * vw.bytes_per_item());
        fin.read(reinterpret_cast<char*>(vb.data_ptr<void>()),         vb.numel()         * vb.bytes_per_item());
        fin.read(reinterpret_cast<char*>(attn_projw.data_ptr<void>()), attn_projw.numel() * attn_projw.bytes_per_item());
        fin.read(reinterpret_cast<char*>(attn_projb.data_ptr<void>()), attn_projb.numel() * attn_projb.bytes_per_item());
        fin.read(reinterpret_cast<char*>(ln_1w.data_ptr<void>()),      ln_1w.numel()      * ln_1w.bytes_per_item());
        fin.read(reinterpret_cast<char*>(ln_1b.data_ptr<void>()),      ln_1b.numel()      * ln_1b.bytes_per_item());
        fin.read(reinterpret_cast<char*>(mlp_fcw.data_ptr<void>()),    mlp_fcw.numel()    * mlp_fcw.bytes_per_item());
        fin.read(reinterpret_cast<char*>(mlp_fcb.data_ptr<void>()),    mlp_fcb.numel()    * mlp_fcb.bytes_per_item());
        fin.read(reinterpret_cast<char*>(mlp_projw.data_ptr<void>()),  mlp_projw.numel()  * mlp_projw.bytes_per_item());
        fin.read(reinterpret_cast<char*>(mlp_projb.data_ptr<void>()),  mlp_projb.numel()  * mlp_projb.bytes_per_item());
        fin.read(reinterpret_cast<char*>(ln_2w.data_ptr<void>()),      ln_2w.numel()      * ln_2w.bytes_per_item());
        fin.read(reinterpret_cast<char*>(ln_2b.data_ptr<void>()),      ln_2b.numel()      * ln_2b.bytes_per_item());

        const Linear query(qw, qb, config_.n_ctx);
        const Linear key(kw, kb, config_.n_ctx);
        const Linear value(vw, vb, config_.n_ctx);
        const Linear out_proj(attn_projw, attn_projb, config_.n_ctx);
        const MultiHeadSelfAttn self_attn(query, key, value, out_proj, config_.n_ctx, config_.n_embed, config_.n_head);
        const LayerNorm ln_1(ln_1w, ln_1b, config_.n_ctx);
        const Linear mlp_fc(mlp_fcw, mlp_fcb, config_.n_ctx);
        const Linear mlp_proj(mlp_projw, mlp_projb, config_.n_ctx);
        const LayerNorm ln_2(ln_2w, ln_2b, config_.n_ctx);
        const GELU gelu(config_.n_ctx, config_.n_embed * 4);

        blocks_.push_back(ResidualAttentionBlock(self_attn, ln_1, mlp_fc, mlp_proj, ln_2, gelu, config_.n_ctx, config_.n_embed));
    }
    
    // LN_F
    fin.read(reinterpret_cast<char*>(&segment_name_size), sizeof(segment_name_size));
    segment_name.resize(segment_name_size);
    fin.read(reinterpret_cast<char*>(segment_name.data()), segment_name_size);
    fin.read(reinterpret_cast<char*>(&segment_size), sizeof(segment_size));
    // std::cout << "Reading segment: [" << segment_name << "](" << segment_size << " bytes)\n";

    Tensor ln_fw({(uint32_t)config_.n_embed}, Dtype::Float32);
    Tensor ln_fb({(uint32_t)config_.n_embed}, Dtype::Float32);
    fin.read(reinterpret_cast<char*>(ln_fw.data_ptr<void>()), ln_fw.numel() * ln_fw.bytes_per_item());
    fin.read(reinterpret_cast<char*>(ln_fb.data_ptr<void>()), ln_fb.numel() * ln_fb.bytes_per_item());

    ln_f_ = LayerNorm(ln_fw, ln_fb, config_.n_ctx);

    res_ = Residual(config_.n_ctx, config_.n_embed);

    timer.stop();
    std::cout << "Load time: " << load_time << " ms\n\n";
}
