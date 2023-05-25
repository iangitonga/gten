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

class TensorMemoryPool {
public:
    TensorMemoryPool(int64_t memsize) {
        memptr_ = reinterpret_cast<uint8_t*>(std::malloc(memsize));
        GTEN_ASSERT(memptr_, "Failed to allocate %ld bytes of memory.", memsize);
        mem_capacity_ = memsize;
        std::printf("Allocated %ldMB of memory.\n", memsize / 1000000);
    }

    void* request_mem(int64_t size) {
        GTEN_ASSERT(
            size <= mem_capacity_ - allocated_mem_,
            "Memory pool failed to allocate %ld bytes of memory. cap=%ld, alloc=%ld, Rem=%ld.",
            size, mem_capacity_, allocated_mem_, mem_capacity_ - allocated_mem_);
        void* mem = reinterpret_cast<void*>(memptr_ + allocated_mem_);
        allocated_mem_ += size;
        return mem;
    }

private:
    int64_t mem_capacity_;
    int64_t allocated_mem_{0}; 
    uint8_t* memptr_;
};


class Tensor
{
public:
    Tensor() = default;

    // Construct a tensor with data the given shape and dtype. Memory pool acts as the
    // allocator for the tensor storage.
    Tensor(TensorMemoryPool& pool, std::initializer_list<int> shape, Dtype dtype);

    // Construct a tensor from an external data source. The constructed tensor does not take
    // ownership of the memory referenced by the pointer.
    Tensor(void* data_ptr, std::initializer_list<int> shape, Dtype dtype);

    // Copy-construct tensor from source tensor. Data from source tensor is shared with
    // the new tensor and thus data itself is not copied from source.
    Tensor(const Tensor& rhs) = default;

    // Move-construct a tensor from source tensor.
    Tensor(Tensor&& rhs) = default;

    // Copy-assign tensor from source tensor. Data from source tensor is shared with the
    // new tensor and thus data itself is not copied from source.
    Tensor &operator=(const Tensor& rhs) = default;

    // Move-assign a tensor from source tensor.
    Tensor &operator=(Tensor&& rhs) = default;

    ~Tensor() = default;
    
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
        return (dtype_ == kFloat16) ? 2 : 4;
    }

    bool is_1d() const noexcept {
        return ndims_ == 1;
    }

    bool is_2d() const noexcept {
        return ndims_ == 2;
    }

    int32_t ndims() const noexcept {
        return ndims_;
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
    void resize(std::initializer_list<int> shape) noexcept;

    friend std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);
    void print() const noexcept;

    /// Returns the size of the give dimension.
    int32_t size(int32_t i) const;

    bool shape_is_equal(std::initializer_list<int> shape) const noexcept;

    size_t nbytes() const noexcept {
        return numel_ * itemsize();
    }

private:
    // Pointer to tensor data storage.
    void* data_{nullptr};
    // Capacity of the data pointer allocated storage.
    size_t storage_capacity_{0};
    // Number of elements in the tensor.
    int32_t numel_{0};
    // Number of tensor dimensions.
    int32_t ndims_{0};
    int32_t shape_[3]{0, 0, 0};
    Dtype dtype_{Dtype::Float32};

    int32_t numel_from_shape() const noexcept;
    void set_shape(std::initializer_list<int> shape);
    void print_single(int32_t item_idx, int32_t col_idx, int32_t n_cols) const noexcept;
};

} // namespace gten
