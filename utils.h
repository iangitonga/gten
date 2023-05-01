#pragma once

#include <chrono>
#include <cstdint>
#include <cmath>
#include <iostream>

// TODO: Allow AVX without F16C for FP32 mode.
#if defined(__AVX__) && defined(__F16C__)
#define GTEN_SIMD 1
#define GTEN_AVX 1

#include <immintrin.h>

#else
#define GTEN_SIMD 0
#define GTEN_AVX 0
#endif

namespace gten
{

// Assert that the given boolean is true. If false, print message and terminate program.
// TODO: Replace with C++ 20 __VA_OPT__, __VA_ARGS__ may not work on non-gcc compilers.
#define GTEN_ASSERT(boolean, message, ...)                                              \
    if (!(boolean)) {                                                                   \
        std::fprintf(stderr, "\x1B[1;31m");                                             \
        std::fprintf(stderr, "GTEN ERROR [File `%s` line %d]: ", __FILE__, __LINE__);   \
        std::fprintf(stderr, message, ##__VA_ARGS__);                                   \
        std::fprintf(stderr, "\n");                                                     \
        std::exit(EXIT_FAILURE);                                                        \
    }  


// FUNDAMENTAL SCALAR DATA TYPES.
typedef int32_t Int32;
typedef uint16_t Float16;
typedef float Float32;

// Allows data type information to be stored and passed around as variables because we
// cannot do that with the types themselves.
enum class Dtype { Int32, Float16, Float32 };

// Convenient shorthands for the enum class above.
static Dtype kInt32 = Dtype::Int32;
static Dtype kFloat16 = Dtype::Float16;
static Dtype kFloat32 = Dtype::Float32;


#if GTEN_AVX

// FUNDAMENTAL VECTOR DATA TYPES.
typedef __m256 Vec8_f32;

// FLOATING POINT VECTOR OPERATIONS

// Overloading for fp16 and fp32 inputs pointer types for load and store instructions
// is type-safe because C++ does not perform implicit conversions between different
// pointer types, including void ptr. For instance, a call to vec8_f32_load with an
// Int32* type, will not compile. Such type-safety is important to avoid subtle bugs.

inline Vec8_f32 vec8_f32_load(const Float16* src_ptr) {
    return _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u *)(const_cast<Float16*>(src_ptr))));
}

inline Vec8_f32 vec8_f32_load(const Float32* src_ptr) {
    return _mm256_loadu_ps(const_cast<Float32*>(src_ptr));
}

inline void vec8_f32_store(Vec8_f32 vec, const Float32* dest_ptr) {
    _mm256_storeu_ps(const_cast<Float32*>(dest_ptr), vec);
}

inline void vec8_f32_store(Vec8_f32 vec, const Float16* dest_ptr) {
    return _mm_storeu_si128((__m128i_u *)(const_cast<Float16*>(dest_ptr)), _mm256_cvtps_ph(vec, 0));
}

inline Vec8_f32 vec8_f32_add(Vec8_f32 a, Vec8_f32 b) {
    return _mm256_add_ps(a, b);
}

// Return A * B + C
inline Vec8_f32 vec8_f32_fma(Vec8_f32 a, Vec8_f32 b, Vec8_f32 c) {
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
}

inline Float32 vec8_f32_sum(Vec8_f32 vec) {
    Float32* f = (Float32 *)(&vec);
    return f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7];
}

inline Vec8_f32 vec8_f32_setzero() {
    return _mm256_setzero_ps();
}
#endif


// FLOATING PONT SCALAR OPERATIONS
inline Float32 fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        Float32 as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

inline uint32_t fp32_to_bits(Float32 f) {
    union {
        Float32 as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

inline Float32 fp16_to_fp32(Float16 h) noexcept
{
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

// MAX:  65504.0 (0)[11110]{1111111111}
// MIN: -65504.0 (1)[11110]{1111111111}

inline Float16 fp32_to_fp16(Float32 f) noexcept
{
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

// Floating point conversion functions below are implemented using templates to ensure
// type-safety because C++ performs implicit conversions between integal types silently
// which can lead to subtle bugs if the functions are invoked with inputs of the
// unintended type. With templates, because we explicitly provide and enforce input
// types, we cannot have such bugs.

// Convert floating point value to Float32. Conversion to Float32 is only allowed from
// Float32 or Float16. Attempt to convert from any other type will cause a runtime
// error. I tried "static_assert" to throw compile-time error but it didn't work.
template<typename input_t>
inline Float32 fpcvt_to_fp32(input_t value) noexcept {
    if constexpr(std::is_same<input_t, Float32>::value) {
        return value;
    }
    else if constexpr(std::is_same<input_t, Float16>::value) {
        return fp16_to_fp32(value);
    }
    else {
        GTEN_ASSERT(false, "Conversion to FP32 is only allowed for FP32 and FP16 types.");
        // Just to avoid "no return statement in function returning non-void" error in
        // case we instantiate using a disallowed type.
        return 0;
    }
}

// Convert Float32 value to a given type. The allowed types are Float32 and Float16.
// Attemt to convert to any other type will cause a runtime error.
// I tried "static_assert" to throw compile-time error but it didn't work.
template<typename output_t>
inline output_t fpcvt_from_fp32(Float32 value) noexcept {
    if constexpr(std::is_same<output_t, Float32>::value) {
        return value;
    }
    else if constexpr(std::is_same<output_t, Float16>::value) {
        return fp32_to_fp16(value);
    }
    else {
        GTEN_ASSERT(false, "Conversion from FP32 is only allowed for FP32 and FP16 types.");
        return 0;
    }
}


class Timer
{
public:
    Timer(int64_t* time_tracker)
        : time_tracker_(time_tracker)
    { start_time_ = std::chrono::high_resolution_clock::now(); }
    ~Timer() { stop(); }

    void stop() {
        if (stopped_)
            return;
        auto end_time = std::chrono::high_resolution_clock::now();
        int64_t start = std::chrono::time_point_cast<std::chrono::milliseconds>(start_time_).time_since_epoch().count();
        int64_t end = std::chrono::time_point_cast<std::chrono::milliseconds>(end_time).time_since_epoch().count();
        int64_t duration = end - start;
        *time_tracker_ += duration;
        stopped_ = true;
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    int64_t* time_tracker_;
    bool stopped_ = false;
};

} // namespace gten

