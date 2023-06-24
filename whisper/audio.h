#pragma once

#include <iostream>

#include "gten_types.h"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"


static float* decode_audio(const char* fname, int* out_n_samples)
{
    uint32_t n_channels;
    uint32_t sample_rate;
    drwav_uint64 n_samples; //  Total number of f32 PCM samples. 
    float* sample_data = drwav_open_file_and_read_pcm_frames_f32(
        fname, &n_channels, &sample_rate, &n_samples, NULL);


    GTEN_ASSERT(sample_data, "Error reading from media file %s.", fname);

    const int out_sample_rate = 16000;
    GTEN_ASSERT(
        sample_rate == out_sample_rate,
        "Expected a sample rate of 16000 but got %d instead.",
        sample_rate);
    GTEN_ASSERT(
        n_channels <= 2,
        "Expected num_channels <=2 but got num_channels=%d instead.",
        n_channels);

    if (n_channels > 1) {
        for (uint64_t i = 0; i < n_samples; i++) {
            sample_data[i] = sample_data[2*i] + sample_data[2*i + 1];
        }
    }

    const int audio_len_secs = n_samples / sample_rate;
    const int n_samples_out = audio_len_secs * out_sample_rate;
    *out_n_samples = n_samples_out;
    float* out_sample_data = new float[n_samples_out];

    void* dest = reinterpret_cast<void*>(out_sample_data);
    const void* src = reinterpret_cast<const void*>(sample_data); 
    std::memcpy(dest, src, n_samples_out * sizeof(float));

    drwav_free(sample_data, NULL);

    return out_sample_data;
}
