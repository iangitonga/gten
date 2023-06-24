#pragma once

#include <cstdio>
#include <cmath>
#include <string.h>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <iostream>

#include "gten_types.h"


// Hanning window as computed by pytorch. size=400, mem=1.6Kb.
float HANN_WINDOW[] = {
	0.0000e+00, 6.1691e-05, 2.4673e-04, 5.5507e-04, 9.8664e-04, 1.5413e-03,
    2.2190e-03, 3.0195e-03, 3.9426e-03, 4.9882e-03, 6.1558e-03, 7.4453e-03,
    8.8564e-03, 1.0389e-02, 1.2042e-02, 1.3815e-02, 1.5708e-02, 1.7721e-02,
    1.9853e-02, 2.2103e-02, 2.4472e-02, 2.6957e-02, 2.9560e-02, 3.2278e-02,
    3.5112e-02, 3.8060e-02, 4.1123e-02, 4.4298e-02, 4.7586e-02, 5.0986e-02,
    5.4497e-02, 5.8117e-02, 6.1847e-02, 6.5684e-02, 6.9629e-02, 7.3680e-02,
    7.7836e-02, 8.2096e-02, 8.6460e-02, 9.0925e-02, 9.5491e-02, 1.0016e-01,
    1.0492e-01, 1.0978e-01, 1.1474e-01, 1.1980e-01, 1.2494e-01, 1.3018e-01,
    1.3552e-01, 1.4094e-01, 1.4645e-01, 1.5204e-01, 1.5773e-01, 1.6349e-01,
    1.6934e-01, 1.7528e-01, 1.8129e-01, 1.8738e-01, 1.9355e-01, 1.9979e-01,
    2.0611e-01, 2.1250e-01, 2.1896e-01, 2.2549e-01, 2.3209e-01, 2.3875e-01,
    2.4548e-01, 2.5227e-01, 2.5912e-01, 2.6604e-01, 2.7300e-01, 2.8003e-01,
    2.8711e-01, 2.9424e-01, 3.0143e-01, 3.0866e-01, 3.1594e-01, 3.2326e-01,
    3.3063e-01, 3.3804e-01, 3.4549e-01, 3.5298e-01, 3.6050e-01, 3.6806e-01,
    3.7566e-01, 3.8328e-01, 3.9093e-01, 3.9861e-01, 4.0631e-01, 4.1404e-01,
    4.2178e-01, 4.2955e-01, 4.3733e-01, 4.4513e-01, 4.5295e-01, 4.6077e-01,
    4.6860e-01, 4.7645e-01, 4.8429e-01, 4.9215e-01, 5.0000e-01, 5.0785e-01,
    5.1571e-01, 5.2355e-01, 5.3140e-01, 5.3923e-01, 5.4705e-01, 5.5487e-01,
    5.6267e-01, 5.7045e-01, 5.7822e-01, 5.8596e-01, 5.9369e-01, 6.0139e-01,
    6.0907e-01, 6.1672e-01, 6.2435e-01, 6.3194e-01, 6.3950e-01, 6.4702e-01,
    6.5451e-01, 6.6196e-01, 6.6937e-01, 6.7674e-01, 6.8406e-01, 6.9134e-01,
    6.9857e-01, 7.0576e-01, 7.1289e-01, 7.1997e-01, 7.2700e-01, 7.3396e-01,
    7.4088e-01, 7.4773e-01, 7.5452e-01, 7.6125e-01, 7.6791e-01, 7.7451e-01,
    7.8104e-01, 7.8750e-01, 7.9389e-01, 8.0021e-01, 8.0645e-01, 8.1262e-01,
    8.1871e-01, 8.2472e-01, 8.3066e-01, 8.3651e-01, 8.4227e-01, 8.4796e-01,
    8.5355e-01, 8.5906e-01, 8.6448e-01, 8.6982e-01, 8.7506e-01, 8.8020e-01,
    8.8526e-01, 8.9022e-01, 8.9508e-01, 8.9984e-01, 9.0451e-01, 9.0907e-01,
    9.1354e-01, 9.1790e-01, 9.2216e-01, 9.2632e-01, 9.3037e-01, 9.3432e-01,
    9.3815e-01, 9.4188e-01, 9.4550e-01, 9.4901e-01, 9.5241e-01, 9.5570e-01,
    9.5888e-01, 9.6194e-01, 9.6489e-01, 9.6772e-01, 9.7044e-01, 9.7304e-01,
    9.7553e-01, 9.7790e-01, 9.8015e-01, 9.8228e-01, 9.8429e-01, 9.8618e-01,
    9.8796e-01, 9.8961e-01, 9.9114e-01, 9.9255e-01, 9.9384e-01, 9.9501e-01,
    9.9606e-01, 9.9698e-01, 9.9778e-01, 9.9846e-01, 9.9901e-01, 9.9944e-01,
    9.9975e-01, 9.9994e-01, 1.0000e+00, 9.9994e-01, 9.9975e-01, 9.9944e-01,
    9.9901e-01, 9.9846e-01, 9.9778e-01, 9.9698e-01, 9.9606e-01, 9.9501e-01,
    9.9384e-01, 9.9255e-01, 9.9114e-01, 9.8961e-01, 9.8796e-01, 9.8618e-01,
    9.8429e-01, 9.8228e-01, 9.8015e-01, 9.7790e-01, 9.7553e-01, 9.7304e-01,
    9.7044e-01, 9.6772e-01, 9.6489e-01, 9.6194e-01, 9.5888e-01, 9.5570e-01,
    9.5241e-01, 9.4901e-01, 9.4550e-01, 9.4188e-01, 9.3815e-01, 9.3432e-01,
    9.3037e-01, 9.2632e-01, 9.2216e-01, 9.1790e-01, 9.1354e-01, 9.0907e-01,
    9.0451e-01, 8.9984e-01, 8.9508e-01, 8.9022e-01, 8.8526e-01, 8.8020e-01,
    8.7506e-01, 8.6982e-01, 8.6448e-01, 8.5906e-01, 8.5355e-01, 8.4796e-01,
    8.4227e-01, 8.3651e-01, 8.3066e-01, 8.2472e-01, 8.1871e-01, 8.1262e-01,
    8.0645e-01, 8.0021e-01, 7.9389e-01, 7.8750e-01, 7.8104e-01, 7.7451e-01,
    7.6791e-01, 7.6125e-01, 7.5452e-01, 7.4773e-01, 7.4088e-01, 7.3396e-01,
    7.2700e-01, 7.1997e-01, 7.1289e-01, 7.0576e-01, 6.9857e-01, 6.9134e-01,
    6.8406e-01, 6.7674e-01, 6.6937e-01, 6.6196e-01, 6.5451e-01, 6.4702e-01,
    6.3950e-01, 6.3194e-01, 6.2434e-01, 6.1672e-01, 6.0907e-01, 6.0139e-01,
    5.9369e-01, 5.8596e-01, 5.7822e-01, 5.7045e-01, 5.6267e-01, 5.5487e-01,
    5.4705e-01, 5.3923e-01, 5.3140e-01, 5.2355e-01, 5.1571e-01, 5.0785e-01,
    5.0000e-01, 4.9215e-01, 4.8429e-01, 4.7645e-01, 4.6860e-01, 4.6077e-01,
    4.5295e-01, 4.4513e-01, 4.3733e-01, 4.2955e-01, 4.2178e-01, 4.1404e-01,
    4.0631e-01, 3.9861e-01, 3.9093e-01, 3.8328e-01, 3.7565e-01, 3.6806e-01,
    3.6050e-01, 3.5298e-01, 3.4549e-01, 3.3804e-01, 3.3063e-01, 3.2326e-01,
    3.1594e-01, 3.0866e-01, 3.0143e-01, 2.9424e-01, 2.8711e-01, 2.8003e-01,
    2.7300e-01, 2.6604e-01, 2.5912e-01, 2.5227e-01, 2.4548e-01, 2.3875e-01,
    2.3209e-01, 2.2549e-01, 2.1896e-01, 2.1250e-01, 2.0611e-01, 1.9979e-01,
    1.9355e-01, 1.8738e-01, 1.8129e-01, 1.7528e-01, 1.6934e-01, 1.6349e-01,
    1.5773e-01, 1.5204e-01, 1.4645e-01, 1.4094e-01, 1.3552e-01, 1.3018e-01,
    1.2494e-01, 1.1980e-01, 1.1474e-01, 1.0978e-01, 1.0492e-01, 1.0016e-01,
    9.5491e-02, 9.0925e-02, 8.6460e-02, 8.2096e-02, 7.7836e-02, 7.3680e-02,
    6.9629e-02, 6.5684e-02, 6.1847e-02, 5.8117e-02, 5.4497e-02, 5.0986e-02,
    4.7586e-02, 4.4298e-02, 4.1123e-02, 3.8060e-02, 3.5112e-02, 3.2278e-02,
    2.9560e-02, 2.6957e-02, 2.4472e-02, 2.2103e-02, 1.9853e-02, 1.7721e-02,
    1.5708e-02, 1.3815e-02, 1.2042e-02, 1.0389e-02, 8.8564e-03, 7.4453e-03,
    6.1558e-03, 4.9882e-03, 3.9426e-03, 3.0195e-03, 2.2190e-03, 1.5413e-03,
    9.8664e-04, 5.5507e-04, 2.4673e-04, 6.1691e-05
};

// Computes mel-spectrogram of the raw audio signal.
class AudioPreprocessor {
public:
	AudioPreprocessor() {
		// mem = 5MB.
		stft_out_ = new float[3000 * 201];
		cos_cache_ = new float[400 * 201];
		sin_cache_ = new float[400 * 201];
		padded_sig_ = new float[16000 * 30 + 400];
		mel_filters_ = new float[80 * 201];

		// Load mel filters.

		std::ifstream fin_mf{"assets/mel_filters.bin"};
	    GTEN_ASSERT(fin_mf.is_open(), "Failed to open mel_filters file.");
		fin_mf.read(reinterpret_cast<char*>(mel_filters_), 80*201*4);

		// Init the caches.
		for (int f = 0; f < 201; ++f) {
			for (int t = 0; t < 400; ++t) {
				cos_cache_[f*400 + t] = std::cos( (2.0f * 3.141592653589793f * f * t) / 400.0f );
				sin_cache_[f*400 + t] = std::sin( (2.0f * 3.141592653589793f * f * t) / 400.0f );
			}
		}
	}

	~AudioPreprocessor() {
		delete[] stft_out_;
		delete[] cos_cache_;
		delete[] sin_cache_;
		delete[] padded_sig_;
		delete[] mel_filters_;
	}

	float* get_mel_filters() const {
		return mel_filters_;
	}

	/**
	 * Computes mel spectrogram of the given audio signal. The audio signal is expected
	 * to contain num_samples=(30s * 16000 samples/s).
	*/
	float* compute_mel_spectrogram(const float* sig, int nsamples) {
		// TODO: Assert nsamples.
		const int nfft = 400;
		const int hop_length = 160;

		// Compute magnitudes. shape=[3000, 201]
		float* magnitudes = stft(sig, nsamples, nfft, hop_length, HANN_WINDOW);

		// Load filters: shape=[80, 201]
		float* mel_filters = get_mel_filters();

		// mel_spec = filters @ magnitudes. shape=[80, 3000]
		// Stores the result of matrix multiplication between the mel_filters and the
		// magnitudes. magnitudes is in transposed order so that we have a good cache
		// locality with respect to both matrices when performing the multiplication.
		const int n_mel_spec = 80*3000;
		float* mel_spec = new float[n_mel_spec];
		{
			const int nrows1 = 80;
			const int ncols = 201;
			const int nrows2 = 3000;

			for (int row1 = 0; row1 < nrows1; ++row1) {
				for (int row2 = 0; row2 < nrows2; ++row2) {
					float dotprod = 0.0f;
					for (int i = 0; i < ncols; ++i) {
						dotprod += mel_filters[row1 * ncols + i] * magnitudes[row2 * ncols + i];
					}
					mel_spec[row1 * nrows2 + row2] = dotprod;
				}
			}
		}

		{
			// log_spec = torch.clamp(mel_spec, min=1e-10).log10()
			for (int i = 0; i < n_mel_spec; ++i) {
				if (mel_spec[i] < 1e-10)
					mel_spec[i] = 1e-10;
				mel_spec[i] = std::log10(mel_spec[i]);
			}
		}


		{
			// torch.maximum(log_spec, log_spec.max() - 8.0)
			float mel_spec_max = -std::numeric_limits<float>::infinity();
			for (int i = 0; i < n_mel_spec; ++i) {
				if (mel_spec[i] > mel_spec_max) {
					mel_spec_max = mel_spec[i];
				}
			}

			for (int i = 0; i < n_mel_spec; ++i) {
				if (mel_spec[i] < mel_spec_max - 8.0) {
					mel_spec[i] = mel_spec_max - 8.0;
				}
			}
		}

		// (log_spec + 4.0) / 4.0
		for (int i = 0; i < n_mel_spec; ++i) {
			mel_spec[i] = (mel_spec[i] + 4.0f) * 1.0f/4.0f;
		}

		return mel_spec;
	}

	float* stft(const float* sig, int nsamples, int nfft, int hop_length, const float* window)
	{
		const int nframes = nsamples / hop_length;
		const int nfreqs = nfft / 2 + 1;

		// PAD_ARRAY
		const int pad_size = nfft / 2;
		float* padded_sig = pad_sig(sig, nsamples, pad_size);

		// COMPUTE_STFT
		for (int i = 0; i < nframes; ++i) {
			float* frame_sig = padded_sig + i * hop_length;
			float* out_sig = stft_out_ + nfreqs * i;
			fourier_transform(frame_sig, nfft, window, out_sig);
		}

		return stft_out_;
	}

	// Pad array by pad_size both on the right and left by reflecting. E.g for pad_size=3
	// input:           [1, 2, 3, 4, 5, 6]
    // output: [4, 3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 3]
	float* pad_sig(const float* sig, int nsamples, int pad_size)
	{
		memcpy(padded_sig_+pad_size, sig, nsamples * 4);

		// Pad left
		const int pad_cnt = pad_size + 1;
		for (int i = 1; i < pad_cnt; ++i) {
			padded_sig_[pad_size - i] = sig[i];
		}

		// Pad right.
		const int offset = nsamples - pad_size - 1;
		const int padded_nsamples = nsamples + pad_size + pad_size;
		for (int i = 0; i < pad_size; ++i) {
			padded_sig_[padded_nsamples - 1 - i] = sig[i + offset];
		}

		return padded_sig_;
	}

	void fourier_transform(const float* sig, int nsamples, const float* window, float* out)
	{
		// Number of frequencies for which magnitudes are calculated.
		int n_freqs = nsamples / 2 + 1;

		for (int f = 0; f < n_freqs; ++f) {
			float sum_real = 0.0f;
			float sum_imag = 0.0f;
			for (int t = 0; t < nsamples; ++t) {
				sum_real += window[t] * sig[t] * cos_cache_[f * 400 + t];
				sum_imag += window[t] * sig[t] * -1.0 * sin_cache_[f * 400 + t];
			}
			out[f] = sum_real * sum_real + sum_imag * sum_imag;
		}
	}

private:
	float* stft_out_;
	float* cos_cache_;
	float* sin_cache_;
	float* padded_sig_;
	float* mel_filters_;
};
