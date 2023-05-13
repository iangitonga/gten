#pragma once

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "gten_types.h"

namespace gten
{

class Tensor
{
public:
    Tensor() = default;
    // Construct a tensor of the given shape and dtype with storage allocated but not
    // initialised.
    Tensor(const std::vector<int32_t>& shape, Dtype dtype, Float32 scale = 0.0f, int zerop = 0);

    // Construct a tensor from the given data source with the given shape and dtype.
    // The data is copied from source and thus the tensor does not assume ownership of
    // the data.
    Tensor(const void* data, const std::vector<int32_t>& shape, Dtype dtype);

    // Copy-construct tensor from source tensor. Data from source tensor is shared with
    // the new tensor and thus data is not copied from source.
    Tensor(const Tensor& rhs) noexcept;

    // Move-construct a tensor from source tensor. The source tensor is left in an invalid
    // state after the move operation.
    Tensor(Tensor&& rhs) noexcept;

    // Copy-assign tensor from source tensor. Data from source tensor is shared with the
    // new tensor and thus data is not copied from source.
    Tensor &operator=(const Tensor& rhs) noexcept;

    // Move-assign a tensor from source tensor. The source tensor is left in an invalid
    // state after the move operation.
    Tensor &operator=(Tensor&& rhs) noexcept;

    ~Tensor();
    
    // Get the pointer to internal data buffer.
    template <typename T>
    T* data_ptr() { 
        return static_cast<T*>(data_); 
    }

    template <typename T>
    const T* data_ptr() const { 
        return static_cast<const T*>(data_); 
    }

    Dtype dtype() const noexcept {
        return dtype_;
    }

    // Get the number of bytes that an element in the tensor occupies.
    size_t itemsize() const noexcept {
        if (dtype_ == kQint8)
            return 1;
        return (dtype_ == kFloat16) ? 2 : 4;
    }

    bool is_1d() const noexcept {
        return shape_.size() == 1;
    }

    bool is_2d() const noexcept {
        return shape_.size() == 2;
    }

    int32_t ndims() const noexcept {
        return shape_.size();
    }

    // Get the number of elems in the tensor.
    int32_t numel() const noexcept {
        return numel_;
    }

    // Resize the tensor to have a new shape. The new shape must not be larger than the
    // shape provided when the tensor was created because this function does not
    // reallocate tensor storage.
    // Note: this is not a reshape function because a reshape function can only reshape
    // a tensor if the new and the existing shapes have the same number of elements.
    void resize(const std::vector<int32_t>& shape) noexcept;
    void resize(std::vector<int32_t>&& shape) noexcept;

    friend std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);
    void print() const noexcept;

    /// Returns the size of the give dimension.
    int32_t size(int32_t i) const;

    const std::vector<int32_t>& shape() const noexcept {
        return shape_;
    }

    size_t nbytes() const noexcept {
        return numel_ * itemsize();
    }

    bool is_quantized() {
        return dtype_ == kQint8;
    }

    float scale() const noexcept {
        return scale_;
    }

    int zero_point() const noexcept {
        return zero_point_;
    }

    // Quantize from the given type to Qint8.
    template<typename scalar_t>
    Float32 deq(scalar_t x) {
        // Use of `if constexpr` instead of explicit specialization to overload for
        // different types because it is not allowed in this scope.
        if constexpr(std::is_same<scalar_t, Float32>::value) {
            return x;
        }
        else if constexpr(std::is_same<scalar_t, Qint8>::value) {
            return scale_ * (Float32)((int)x - zero_point_);
        }
        else {
            GTEN_ASSERT(false, "Dequantize only allowed from Float32 and Qint8 types.");
        }
    }

    // Quantize from Float32 to the given type.
    template<typename scalar_t>
    scalar_t qu(Float32 x) {
        if constexpr(std::is_same<scalar_t, Float32>::value) {
            return x;
        }
        else if constexpr(std::is_same<scalar_t, Qint8>::value) {
            return static_cast<Qint8>(std::roundf(x/scale_ + zero_point_));
        }
        else {
            GTEN_ASSERT(false, "Quantize only allowed to Float32 and Qint8 types.");
        }
    }

public:
    static uint64_t total_memory_allocated;

private:
    std::vector<int32_t> shape_;
    std::vector<int32_t> strides_;
    Dtype dtype_;
    int32_t numel_{0};

    // Allocated storage size, in bytes, when the tensor is initialised.
    size_t storage_size_{0};

    // Pointer to tensor data storage.
    void* data_;

    // We use a simple reference counting technique to allow many tensors to share the
    // same data. When tensors are copied, we do not allocate new memory for the new copy
    // and copy the data. Instead, we simply copy the data pointer along with the
    // heap-allocated refcount pointer so that the two tensors share the same data. When
    // data is copied we increase the refcount and decrease it when a tensor pointing to
    // the same data is destroyed. If the refcount gets to zero, we deallocate the data.
    // Note: Using `std::shared_ptr` could probably have been more appropriate here.
    int32_t* refcount_{nullptr};

    // Params for quantized tensors, i.e tensors with dtype=Qint8, to allow quantization
    // and dequantization.
    Float32 scale_{0.0f}; 
    int32_t zero_point_{0}; 

    // Increase refcount by one.
    void incref();

    // Decrease refcount by one. If refcount hits zero, the data is deallocated.
    void decref();

    void set_strides(const std::vector<int32_t>& shape);
    int32_t numel_from_shape() const noexcept;
    void validate_shape(const std::vector<int32_t>& shape) const;
    void print_single(int32_t item_idx, int32_t col_idx, int32_t n_cols) const noexcept;
};

} // namespace gten
