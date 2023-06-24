#include <fstream>

#include "gten.h"
#include "tokenizer.h"

#include "stft.h"
#include "audio.h"


using namespace gten;

class AudioEncoder {
public:
    AudioEncoder(int in_channels, int in_frames, int n_blocks, int attn_heads, int d_embed, int n_ctx)
        : conv1{Conv1d(d_embed, in_channels, in_frames, /*strides=*/1)}, 
          conv2{Conv1d(d_embed, d_embed, in_frames, /*strides=*/2)},
          pos_emb{PosEmbedding(n_ctx, d_embed)},
          res{Residual(n_ctx, d_embed)},
          gelu1{GELU(d_embed, in_frames, false)},
          gelu2{GELU(n_ctx, d_embed, false)},
          ln{LayerNorm(n_ctx, d_embed)}
    {
        blocks.reserve(n_blocks);
        for (int i = 0; i < n_blocks; i++) {
            blocks.push_back(ResidualAttnBlock(attn_heads, d_embed, d_embed*4, n_ctx, /*mask_attn=*/false));
        }
    }

    Tensor forward(const Tensor& inp)
    {
        // inp: [in_channels, nframes] ->
        // inp: [80, 3000] -> [384, 3000]
        Tensor out = gelu1.forward(conv1.forward(inp));
        // [384, 3000] -> [1500, 384]
        out = gelu2.forward(conv2.forward(out));
        // // Expected out: [nframes, n_filters]
        out = res.forward(out, pos_emb.forward(1500));
        // Tensor out0 = blocks[0].attn.forward(blocks[0].attn_ln.forward(out));
        // print_head(out0);
        for (auto& block : blocks)
            out = block.forward(out);
        out = ln.forward(out);
        return out;
    }

    void reset_caches() {
        res.reset_acv_cache();
        gelu1.reset_acv_cache();
        gelu2.reset_acv_cache();
        ln.reset_acv_cache();
        for (auto& block : blocks)
            block.reset_acv_cache();
    }

    void show_performance() {
        int64_t block_lin = 0;
        int64_t block_attn = 0;
        for (const auto& block : blocks) {
            block_lin += block.time_linear();
            block_attn += block.time_attn();
        }

        std::cout << "conv1: " << conv1.exec_time()  << "\n"
                  << "gelu1: " << gelu1.time()  << "\n"
                  << "conv2: " << conv2.exec_time()  << "\n"
                  << "gelu2: " << gelu2.time()  << "\n"
                  << "posem: " << pos_emb.time()  << "\n"
                  << "res  : " << res.time()  << "\n"
                  << "ln   : " << res.time()  << "\n"
                  << "blckl: " << block_lin << ", perb=" << block_lin/4 << "\n"
                  << "blcka: " << block_attn << ", perb=" << block_attn/4  << "\n";
    }

public:
    Conv1d conv1;
    Conv1d conv2;
    PosEmbedding pos_emb;
    Residual res;
    GELU gelu1;
    GELU gelu2;
    std::vector<ResidualAttnBlock> blocks;
    LayerNorm ln;
};

// TODO: ASSERT B4 VECTOR

class Decoder {
public:
    Decoder(int n_vocab, int n_blocks, int attn_heads, int d_embed, int n_q_ctx, int n_kv_ctx)
        : token_emb{Embedding(n_vocab, d_embed, n_q_ctx)},
          pos_emb{PosEmbedding(n_q_ctx, d_embed)},
          ln{LayerNorm(n_q_ctx, d_embed)},
          res1{Residual(n_q_ctx, d_embed)}
    {
        blocks.reserve(n_blocks);
        for (int i = 0; i < n_blocks; i++)
        {
            blocks.push_back(ResidualCrossAttnBlock(attn_heads, d_embed, d_embed*4, n_q_ctx, n_kv_ctx));
        }
        
    }

    Tensor forward(const Tensor& x_, const Tensor& xa)
    {
        Tensor out = res1.forward(token_emb.forward(x_), pos_emb.forward(x_.numel()));
        for (auto& block : blocks)
        {
            out = block.forward(out, xa);
        }
        out = ln.forward(out);
        out = token_emb.forward_proj(out);

        return out;
    }

    void reset_caches() {
        token_emb.reset_acv_cache();
        for (auto& block : blocks)
        {
            block.reset_acv_cache();
        }
        ln.reset_acv_cache();
        res1.reset_acv_cache();
    }

public:
    Embedding token_emb;
    PosEmbedding pos_emb;
    std::vector<ResidualCrossAttnBlock> blocks;
    LayerNorm ln;
    Residual res1;
};

class WhisperConfig {
public:
    int n_mels;
    int n_vocab;
    int n_audio_ctx;
    int n_audio_state;
    int n_audio_head;
    int n_audio_layer;
    int n_text_ctx;
    int n_text_state;
    int n_text_head;
    int n_text_layer;

    friend std::ostream& operator<<(std::ostream& stream, const WhisperConfig& cfg)
    {
        std::cout << "WhisperConfig:\n"
                  << "  n_mels: "        << cfg.n_mels        << "\n"
                  << "  n_vocab: "       << cfg.n_vocab       << "\n"
                  << "  n_audio_layer: " << cfg.n_audio_layer << "\n"
                  << "  n_audio_ctx: "   << cfg.n_audio_ctx   << "\n"
                  << "  n_audio_state: " << cfg.n_audio_state << "\n"
                  << "  n_audio_head: "  << cfg.n_audio_head  << "\n"
                  << "  n_text_layer: "  << cfg.n_text_layer  << "\n"
                  << "  n_text_ctx: "    << cfg.n_text_ctx    << "\n"
                  << "  n_text_state: "  << cfg.n_text_state  << "\n"
                  << "  n_text_head: "   << cfg.n_text_head   << "\n";

        return stream;
    }
};


void load_weight(std::ifstream& fin, Tensor& dest, const std::string& expected_name)
{
    int tensor_name_size;
    std::string tensor_name;

    fin.read(reinterpret_cast<char*>(&tensor_name_size), sizeof(tensor_name_size));
    tensor_name.resize(tensor_name_size);
    fin.read(tensor_name.data(), tensor_name_size);

    GTEN_ASSERT(
        tensor_name == expected_name,
        "Load error: expected tensor %s but got %s.",
        expected_name.c_str(),
        tensor_name.c_str()
    );

    int tensor_nbytes;
    fin.read(reinterpret_cast<char*>(&tensor_nbytes), sizeof(tensor_nbytes));

    GTEN_ASSERT(
        static_cast<size_t>(tensor_nbytes) == dest.nbytes(),
        "Load error: expected tensor to have size %ld but got %d.",
        dest.nbytes(),
        tensor_nbytes
    );

    fin.read(dest.data_ptr<char>(), dest.nbytes());
}

// All non-language tokens that are suppressed during decoding.
static const int s_ENGLISH_VOCAB_BAD_TOKENS[98] = {
    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92,
    93, 357, 366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267, 1279, 1303, 1343, 1377,
    1391, 1635, 1782, 1875, 1906, 2162, 2361, 2488, 3467, 3880, 4008, 4211, 4600, 4808, 5299,
    5855, 6329, 7203, 8864, 9609, 9959, 10221, 10563, 10786, 11420, 11709, 11907, 13163,
    13697, 13700, 14808, 15306, 16410, 16791, 17174, 17992, 19203, 19510, 20368, 20724,
    22305, 22935, 23090, 27007, 29113, 30109, 30420, 30906, 33409, 34949, 40283, 40493, 
    40549, 41906, 46111, 47282, 49146, 49704, /*no_speech*/50361, /*no_timestamps*/50362
};

void apply_timestamp_rules(const std::vector<int>& tokens, Tensor& logits, const Tokenizer& tokenizer)
{
    const int sot_seq_len = 1; // tokenizer.prompt_length()

    // Rule: Timestamps must appear in pairs except for the first timestamp and the last.
    // The rule applied below is if the previous two predictions were timestamps, we
    // mask all the timestamps. If only the previous prediction is a timestamp, we mask
    // all the language tokens.
    const int last_token_idx = tokens.size() - 1;
    // [a, b, c, t0, ]
    const int n_logits = logits.numel();
    float* logits_data = logits.data_ptr<float>();

    const int n_pred = tokens.size() - sot_seq_len;
    bool last_was_timestamp = false;
    if (n_pred >= 1 && tokens[last_token_idx] >= tokenizer.timestamp_begin) {
        last_was_timestamp = true;
    }
    bool penultimate_was_timestamp = false;
    if (n_pred < 2 || tokens[last_token_idx - 1] >= tokenizer.timestamp_begin) {
        penultimate_was_timestamp = true;
    }
    if (last_was_timestamp) {
        if (penultimate_was_timestamp) {
            for (int i = tokenizer.timestamp_begin; i < n_logits; ++i) {
                logits_data[i] = -INFINITY;
            }
        } else {
            for (int i = 0; i < tokenizer.eot; ++i) {
                logits_data[i] = -INFINITY;
            }
        }
    }


    const float precision = 30.0f / 448.0f;
    // Initial timestamp cannot be later than this.
    const float max_initial_timestamp = 1.0f;
    const int max_initial_timesamp_index = std::round(max_initial_timestamp / precision);
    if (tokens.size() == sot_seq_len) {
        int last_allowed = tokenizer.timestamp_begin + max_initial_timesamp_index;
        for (int i = last_allowed + 1; i < logits.numel(); ++i) {
            logits_data[i] = -INFINITY;
        }
    }

    // If sum of probs of timestamps is higher than that of any other token, sample ts.
    float sum_probs_timestamps = 0.0f;
    for (int i = tokenizer.timestamp_begin; i < logits.numel(); ++i) {
         sum_probs_timestamps += logits_data[i];
    }
    bool sum_probs_timestamps_is_higher = true;
    for (int i = 0; i < tokenizer.timestamp_begin; ++i) {
        if (logits_data[i] > sum_probs_timestamps) {
            sum_probs_timestamps_is_higher = false;
            break;
        }
    }

    if (sum_probs_timestamps_is_higher) {
        for (int i = 0; i < tokenizer.timestamp_begin; ++i)
        {
            logits_data[i] = -INFINITY;
        }
    }
}

static void print_usage() {
    std::cout << "Usage:\n ./whisper [options] <FILEPATH>" << "\n";
    std::cout << "FILEPATH is a path to a wav file with samplerate 16000hz";
    std::cout << "Options include:\n"
              << "  -tiny for tiny English model.\n"
              << "  -base for base English model.\n"
              << "  -small for small English model.\n"
              << "  -medium for medium English model.\n";
}

int main(int argc, const char *argv[])
{
    if (argc < 2) {
        print_usage();
        return -1;
    }

    std::string model_name = "base";
    std::string media_fpath;
    if (argc > 2)
    {
        media_fpath = argv[2];
        std::string model_arg{argv[1]};
        if (model_arg == "-tiny") {
            model_name = "tiny";
        } else if (model_arg == "-base") {
            model_name = "base";
        } else if (model_arg == "-small") {
            model_name = "small";
        } else if (model_arg == "medium") {
            model_name = "medium";
        } else if (model_arg == "--help") {
            print_usage();
            return 0;
        } else {
            std::cout << "Unknown option: " << model_arg << "\n";
            print_usage();
            return -1;
        }
    } else {
        std::string path_arg{argv[1]};
        media_fpath = path_arg;
        if (path_arg == "--help") {
            print_usage();
            return 0;
        }
    }

#if defined(__WIN32__) || defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
    std::string dl_command = std::string("python model_dl.py ") + model_name;
#else
    std::string dl_command = std::string("python3 model_dl.py ") + model_name;
#endif

    int res = std::system(dl_command.c_str());
    if (res != 0) {
        std::cout << "Error: Failed to download " << model_name << " model due to network issues.\n";
        return -1;
    }

    std::ifstream fin{"models/whisper.tiny.en.gten", std::ios::binary};
    GTEN_ASSERT(fin.is_open(), "model file is missing.");

    int64_t magic;
    fin.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    GTEN_ASSERT(magic == 0x454c49464e455447, "Magic number in the binary does not match the expected one.\n");

    WhisperConfig model_config;
    fin.read(reinterpret_cast<char*>(&model_config), sizeof(model_config));
    std::cout << model_config << "\n";

    AudioEncoder enc{model_config.n_mels, 3000,
        model_config.n_audio_layer, model_config.n_audio_head,
        model_config.n_audio_state, model_config.n_audio_ctx};

    load_weight(fin, enc.conv1.weight, "enc.conv1.w");
    load_weight(fin, enc.conv1.bias, "enc.conv1.b");
    load_weight(fin, enc.conv2.weight, "enc.conv2.w");
    load_weight(fin, enc.conv2.bias, "enc.conv2.b");
    load_weight(fin, enc.pos_emb.weight, "enc.pos_emb.w");

    for (int i = 0; i < model_config.n_audio_layer; i++)
    {
        auto& block = enc.blocks[i];
        std::string block_name = "enc.b" + std::to_string(i);
        load_weight(fin, block.attn.query.weight, block_name + ".attn.query.w");
        load_weight(fin, block.attn.query.bias, block_name + ".attn.query.b");
        load_weight(fin, block.attn.key.weight, block_name + ".attn.key.w");
        load_weight(fin, block.attn.key.bias, block_name + ".attn.key.b");
        load_weight(fin, block.attn.value.weight, block_name + ".attn.value.w");
        load_weight(fin, block.attn.value.bias, block_name + ".attn.value.b");
        load_weight(fin, block.attn.qkv_proj.weight, block_name + ".attn.qkv_proj.w");
        load_weight(fin, block.attn.qkv_proj.bias, block_name + ".attn.qkv_proj.b");
        load_weight(fin, block.attn_ln.weight, block_name + ".attn_ln.w");
        load_weight(fin, block.attn_ln.bias, block_name + ".attn_ln.b");
        load_weight(fin, block.mlp_fc.weight, block_name + ".mlp_fc.w");
        load_weight(fin, block.mlp_fc.bias, block_name + ".mlp_fc.b");
        load_weight(fin, block.mlp_proj.weight, block_name + ".mlp_proj.w");
        load_weight(fin, block.mlp_proj.bias, block_name + ".mlp_proj.b");
        load_weight(fin, block.mlp_ln.weight, block_name + ".mlp_ln.w");
        load_weight(fin, block.mlp_ln.bias, block_name + ".mlp_ln.b");
    }

    load_weight(fin, enc.ln.weight, "enc.ln.w");
    load_weight(fin, enc.ln.bias, "enc.ln.b");

    Decoder dec{model_config.n_vocab, model_config.n_text_layer, model_config.n_text_head,
                model_config.n_text_state, model_config.n_text_ctx, model_config.n_audio_ctx};
    load_weight(fin, dec.token_emb.weight, "dec.emb.w");
    load_weight(fin, dec.pos_emb.weight, "dec.pos_emb.w");

    for (int i = 0; i < model_config.n_text_layer; i++)
    {
        auto& block = dec.blocks[i];
        std::string block_name = "dec.b" + std::to_string(i);
        load_weight(fin, block.attn.query.weight, block_name + ".attn.query.w");
        load_weight(fin, block.attn.query.bias, block_name + ".attn.query.b");
        load_weight(fin, block.attn.key.weight, block_name + ".attn.key.w");
        load_weight(fin, block.attn.key.bias, block_name + ".attn.key.b");
        load_weight(fin, block.attn.value.weight, block_name + ".attn.value.w");
        load_weight(fin, block.attn.value.bias, block_name + ".attn.value.b");
        load_weight(fin, block.attn.qkv_proj.weight, block_name + ".attn.qkv_proj.w");
        load_weight(fin, block.attn.qkv_proj.bias, block_name + ".attn.qkv_proj.b");
        load_weight(fin, block.attn_ln.weight, block_name + ".attn_ln.w");
        load_weight(fin, block.attn_ln.bias, block_name + ".attn_ln.b");
        load_weight(fin, block.cross_attn.query.weight, block_name + ".cross_attn.query.w");
        load_weight(fin, block.cross_attn.query.bias, block_name + ".cross_attn.query.b");
        load_weight(fin, block.cross_attn.key.weight, block_name + ".cross_attn.key.w");
        load_weight(fin, block.cross_attn.key.bias, block_name + ".cross_attn.key.b");
        load_weight(fin, block.cross_attn.value.weight, block_name + ".cross_attn.value.w");
        load_weight(fin, block.cross_attn.value.bias, block_name + ".cross_attn.value.b");
        load_weight(fin, block.cross_attn.qkv_proj.weight, block_name + ".cross_attn.qkv_proj.w");
        load_weight(fin, block.cross_attn.qkv_proj.bias, block_name + ".cross_attn.qkv_proj.b");
        load_weight(fin, block.cross_attn_ln.weight, block_name + ".cross_attn_ln.w");
        load_weight(fin, block.cross_attn_ln.bias, block_name + ".cross_attn_ln.b");
        load_weight(fin, block.mlp_fc.weight, block_name + ".mlp_fc.w");
        load_weight(fin, block.mlp_fc.bias, block_name + ".mlp_fc.b");
        load_weight(fin, block.mlp_proj.weight, block_name + ".mlp_proj.w");
        load_weight(fin, block.mlp_proj.bias, block_name + ".mlp_proj.b");
        load_weight(fin, block.mlp_ln.weight, block_name + ".mlp_ln.w");
        load_weight(fin, block.mlp_ln.bias, block_name + ".mlp_ln.b");
    }

    load_weight(fin, dec.ln.weight, "dec.ln.w");
    load_weight(fin, dec.ln.bias, "dec.ln.b");

    int num_samples;
    float* decode_data = decode_audio(media_fpath.c_str(), &num_samples);
    const int samplerate = 16000;
    const int audio_chunk_frames = 3000;
    Tensor audio({num_samples + samplerate*26}, kFloat32);
    float* audio_data = audio.data_ptr<float>();
    std::memset(audio_data, 0, audio.nbytes()); // ensure padded bytes are 0.

    float audio_max = -INFINITY;
    for (int i = 0; i < num_samples; i++)
    {
        float val = decode_data[i];
        audio_data[i] = val;
        if (val > audio_max)
            audio_max = val;
    }
    for (int i = 0; i < num_samples; i++) {
        audio_data[i] = audio_data[i] / audio_max;
    }
    
    AudioPreprocessor preproc;
    Tokenizer tokenizer;

    const int segment_samples = samplerate * 30 ;
    float segment_time_offset = 0;

    std::cout << "\nTRANSCRIPTION: \n\n";
    while (segment_time_offset*samplerate + segment_samples <= audio.numel())
    {
        float* data_ptr = audio.data_ptr<float>() + static_cast<int>(segment_time_offset) * samplerate;
        float* mel_data = preproc.compute_mel_spectrogram(data_ptr, segment_samples);
        Tensor mel(mel_data, {model_config.n_mels, audio_chunk_frames}, kFloat32);

        enc.reset_caches();
        Tensor xa = enc.forward(mel);

        std::vector<int> tokens = {tokenizer.sot};
        tokens.reserve(model_config.n_text_ctx/2);

        std::vector<int> print_tokens;
        print_tokens.reserve(64);

        dec.reset_caches();
        bool is_begin_ts = true;
        float begin_ts = 0;
        for (int j_ = 0; j_ < model_config.n_text_ctx/2; ++j_)
        {
            Tensor x{tokens.data(), {(int)tokens.size()}, kInt32};
            Tensor logits = dec.forward(x, xa);
            float* logits_data = logits.data_ptr<float>();

            { // SOFTMAX
                float max = -INFINITY;
                for (int i = 0; i < logits.numel(); ++i) {
                    if (logits_data[i] > max)
                        max = logits_data[i];
                }

                float sum_exp = 0.0f;
                for (int i = 0; i < logits.numel(); i++) {
                    float xd = logits_data[i];
                    float exp_val = std::exp(xd - max);
                    logits_data[i] = exp_val;
                    sum_exp += exp_val;
                }

                for (int i = 0; i < logits.numel(); i++) {
                    float xd = logits_data[i];
                    logits_data[i] = xd / sum_exp;
                }
            }

            apply_timestamp_rules(tokens, logits, tokenizer);

            int pred_token = 0;
            { // SUPPRESS FORBIDDEN TOKENS.
                float max = -INFINITY;
                for (int k = 0; k < logits.numel(); ++k) {
                    float val = logits_data[k];
                    for (int l = 0; l < 98; ++l) {
                        if (k == s_ENGLISH_VOCAB_BAD_TOKENS[l]) {
                            val = -INFINITY;
                            break;
                        }
                    }
                    // Select argmax.
                    if (val > max) {
                        max = val;
                        pred_token = k;
                    }
                }
            }

            if (pred_token == tokenizer.eot)
                break;

            if (tokenizer.is_timestamp(pred_token)) {
                const float ts = tokenizer.decode_ts_token(pred_token);
                if (is_begin_ts) {
                    begin_ts = ts;
                    is_begin_ts = false;
                } else {
                    const float time_offset_secs = segment_time_offset;
                    const int begin_time_tot_secs = static_cast<int>(std::round(begin_ts + time_offset_secs));
                    const int begin_time_mins = begin_time_tot_secs / 60;
                    const int begin_time_secs = begin_time_tot_secs - (begin_time_mins * 60);

                    const int end_time_tot_secs = static_cast<int>(std::round(ts + time_offset_secs));
                    const int end_time_mins = end_time_tot_secs / 60;
                    const int end_time_secs = end_time_tot_secs - (end_time_mins*60);

                    std::cout << '[' 
                              << std::setfill('0') << std::setw(2) << begin_time_mins
                              << ':'
                              << std::setfill('0') << std::setw(2) << begin_time_secs
                              << " - "
                              << std::setfill('0') << std::setw(2) << end_time_mins
                              << ':'
                              << std::setfill('0') << std::setw(2) << end_time_secs
                              << ']';

                    for (int token : print_tokens)
                        std::cout << tokenizer.decode_token(token);
                    std::cout << '\n';

                    print_tokens.clear();
                    is_begin_ts = true;
                }
            } else {
                print_tokens.push_back(pred_token);
            }

            tokens.push_back(pred_token);
        }
        std::cerr << "\n";

        segment_time_offset = segment_time_offset + begin_ts;
    }

    return 0;
}
