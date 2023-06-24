#pragma once

#include <iostream>
#include <string>


class Tokenizer {
public:
    int sot{50257};
    int eot{50256};
    int transcribe{50358};
    int translate{50357};
    int no_speech{50361};
    int no_timestamps{50362};
    int timestamp_begin{50363};
    int timestamp_end{51863};
    int english_token{50258};

    Tokenizer() {
        vocab_ = new char[vocab_bufsize_];
        FILE* vocab_file = std::fopen(m_vocab_filepath.c_str(), "r");
        if (!vocab_file) {
            std::cerr << "Failed to open vocab file\n";
            std::exit(-1);
        }

        int char_hold;
        for (int token_offset = 0; token_offset < vocab_bufsize_; token_offset += word_size_) {
            int char_idx = 0;
            while((char_hold = std::fgetc(vocab_file)) != EOF
                   && char_hold != '\n'
                   && char_idx < word_size_ - 1) {
            vocab_[token_offset + char_idx] = (char)char_hold;
            char_idx += 1;
            }
            vocab_[token_offset + char_idx] = '\0';
        }
        std::fclose(vocab_file);
    }
    
    std::string decode_token(int token) const {
        if (token < 50256) {
            return std::string(vocab_ + (token * word_size_));
        } else if (is_timestamp(token)) {
            float ts = static_cast<float>(token - timestamp_begin) * 0.02f;
            std::string out = std::to_string(ts);
            out.resize(4);
            out = std::string("<") + out + std::string(">");
            return out;
        }
        else {
            // std::cerr << "Unknown token: " << token << "\n";
            return std::string("<UNK>");
        }
        // float ts = (token - timestamp_begin) * 0.02f;
    }

    float decode_ts_token(int token) const {
        return (token - timestamp_begin) * 0.02f;
    }

    bool is_timestamp(int token) const
    {
      return token >= timestamp_begin;
    }

private:
    // Path to the vocabulary file.
    std::string m_vocab_filepath{"assets/vocab.txt"};

    // Memory size, in bytes, allocated for each word.
    const int word_size_ = 50;

    // Total number of words in the vocabulary. 
    int vocab_size_{50256};

    // Size of the vocab buffer, in bytes.
    int vocab_bufsize_{50256 * 50};

    // The buffer (2MB size) which holds all the vocab words. Conceptually, it contains
    // fixed-size slots for each word in the vocabulary. The slot size is large enough to 
    // hold the longest word.
    // When creating the vocabulary, we read each word from a file, add it to its slot in
    // the buffer and then add a null-termination character so that for words smaller than
    // the slots, we know where they end. The words are stored in the order of their
    // corresponding token value so the first slot contains the word that when decoded,
    // maps to token value 0. That allows for very fast decoding because we can index a
    // word directly by adding an offset to to the buffer pointer. We could potentially
    // use something like hash map to make it much simpler but we would have to allocate 
    // memory for each slot independently which would lead to bad memory fragmentation.
    char *vocab_;
};