#pragma once

#include <cstdint>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <vector>


// Replace with C++ 20 __VA_OPT__, ##__VA_ARGS__ may not work on non-gcc compilers. 
#define GTEN_CHECK(boolean, message, ...)             \
    if (!(boolean)) {                                 \
        std::fprintf(stderr, "LINE: %d: ", __LINE__); \
        std::fprintf(stderr, message, ##__VA_ARGS__); \
        std::fprintf(stderr, "\n");                   \
        throw std::runtime_error("gten error");       \
    }  

namespace gten {

enum class Dtype { Float32, Int32 };

class Tensor
{
public:
    Tensor() {};
    // Construct a tensor of the given shape and dtype with storage allocated but not initialised.
    Tensor(const std::vector<uint32_t> &shape, Dtype dtype = Dtype::Float32);

    // Construct a tensor from the given data source with the given shape and dtype.
    // The tensor copies data and thus does assume ownership of the data.
    Tensor(const void *data, const std::vector<uint32_t> &shape, Dtype dtype);

    // Copy-construct tensor from source tensor. Data from source tensor is shared with
    // the new tensor and thus data is not copied from source.
    Tensor(const Tensor &rhs) { copy_from_other(rhs); };

    // Copy-assign tensor from source tensor. Data from source tensor is shared with the
    // new tensor and thus data is not copied from source.
    Tensor &operator=(const Tensor &rhs) { copy_from_other(rhs); return *this; };

    // Move-construct a tensor from source tensor. The source tensor is left in an invalid
    // state after the move operation.
    Tensor(Tensor &&rhs) { move_from_other(std::move(rhs)); };

    // Move-assign a tensor from source tensor. The source tensor is left in an invalid
    // state after the move operation.
    Tensor &operator=(Tensor &&rhs) { move_from_other(std::move(rhs)); return *this; };

    size_t bytes_per_item() const { return 4; };
    template <typename T> T *data_ptr() { return static_cast<T*>(m_data.get()); };
    template <typename T> const T *data_ptr() const { return static_cast<T*>(m_data.get()); };
    Dtype dtype() const { return m_dtype; };
    bool is_1d() const { return m_shape.size() == 1; };
    bool is_2d() const { return m_shape.size() == 2; };
    uint64_t ndims() const { return m_shape.size(); };
    // Return number of elems in the tensor.
    uint32_t numel() const { return m_numel; };
    void reshape(const std::vector<uint32_t> &shape) { validate_shape(shape); m_shape = shape; m_numel = numel_from_shape(); }
    friend std::ostream& operator<<(std::ostream& stream, const Tensor& tensor) { tensor.print();return stream; };
    void print() const;
    Tensor to(const Dtype dtype);
    const std::vector<uint32_t> &shape() const { return m_shape; };

    /// Returns the size of dimension of the given index.
    uint32_t size(uint32_t i) const {
        GTEN_CHECK(i < ndims(), "Tensor dim access, %d, is out of range of a tensor with %ld-dims.", i, ndims());
        return m_shape[i];
    }

private:
    std::vector<uint32_t> m_shape;
    std::vector<uint32_t> m_strides;
    Dtype m_dtype = Dtype::Float32;

    uint32_t m_numel;
    // We may share the same data across many tensors.
    std::shared_ptr<void> m_data;

    void set_strides(const std::vector<uint32_t> &shape);
    uint32_t numel_from_shape();
    // Note: This method does NOT copy the data in the buffer from other to self. It
    // only copies the pointer. If either `self` or `other` data is modified after a
    // call to this method, the modification will affect both tensors.
    void copy_from_other(const Tensor &other);
    void move_from_other(Tensor &&other);
    void validate_shape(const std::vector<uint32_t> &shape) const;
    void print_single(const uint32_t item_idx, const uint32_t col_idx, const uint32_t n_cols) const;
};

} // namespace gten
