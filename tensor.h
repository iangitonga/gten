#pragma once

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "gten_types.h"

namespace gten {

class Tensor
{
public:
    Tensor() {};
    // Construct a tensor of the given shape and dtype with storage allocated but not
    // initialised.
    Tensor(const std::vector<int32_t>& shape, Dtype dtype);

    // Construct a tensor from the given data source with the given shape and dtype.
    // The data is copied from source and thus the tensor does not assume ownership of
    // the data.
    Tensor(const void* data, const std::vector<int32_t>& shape, Dtype dtype);

    // Copy-construct tensor from source tensor. Data from source tensor is shared with
    // the new tensor and thus data is not copied from source.
    Tensor(const Tensor& rhs) noexcept;

    // Copy-assign tensor from source tensor. Data from source tensor is shared with the
    // new tensor and thus data is not copied from source.
    Tensor &operator=(const Tensor& rhs) noexcept;

    // Move-construct a tensor from source tensor. The source tensor is left in an invalid
    // state after the move operation.
    Tensor(Tensor&& rhs) noexcept;

    // Move-assign a tensor from source tensor. The source tensor is left in an invalid
    // state after the move operation.
    Tensor &operator=(Tensor&& rhs) noexcept;

    ~Tensor();
    
    // Get the pointer to internal data buffer.
    template <typename T> T* data_ptr() { return static_cast<T*>(data_); };
    template <typename T> const T* data_ptr() const { return static_cast<T*>(data_); };
    Dtype dtype() const noexcept;

    // Get the number of bytes that an element in the tensor occupies.
    size_t itemsize() const noexcept;

    bool is_1d() const noexcept;
    bool is_2d() const noexcept;
    int32_t ndims() const noexcept;

    // Get the number of elems in the tensor.
    int32_t numel() const noexcept;

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
    const std::vector<int32_t>& shape() const noexcept;
    size_t nbytes() const noexcept;

private:
    std::vector<int32_t> shape_;
    std::vector<int32_t> strides_;
    Dtype dtype_;
    int32_t numel_ = 0;

    // Allocated storage size, in bytes, when the tensor is initialised.
    size_t storage_size_ = 0;

    // Pointer to tensor data storage.
    void* data_;

    // We use a simple reference counting technique to allow many tensors to share the
    // same data. This pointer points to a heap-allocated block where we record how many
    // tensors share the same data as this tensor. Each of those tensors also have a copy
    // of this pointer. When a tensor is copied, the refcount is increased. When a tensor
    // destructor is called, it decreases the refcount. If the refcount hits zero the
    // data get deallocated and the refcount is set to nullptr.
    // Note: The current refcount implementation is NOT thread-safe.
    int32_t* refcount_ = nullptr;

    // Increase refcount by one.
    void incref();

    // Decrease refcount by one. If refcount hits zero, the data is deallocated.
    void decref();

    void set_strides(const std::vector<int32_t>& shape);
    int32_t numel_from_shape() const noexcept;
    // Note: This method does NOT copy the data in the buffer from other to self. It
    // only copies the pointer. If either `self` or `other` data is modified after a
    // call to this method, the modification will affect both tensors.
    void validate_shape(const std::vector<int32_t>& shape) const;
    void print_single(int32_t item_idx, int32_t col_idx, int32_t n_cols) const noexcept;
};

} // namespace gten
