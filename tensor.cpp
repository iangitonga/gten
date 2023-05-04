#include "tensor.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>


void* alloc_mem(size_t nbytes) noexcept
{
    void* ptr = std::malloc(nbytes);
    GTEN_ASSERT(ptr, "Memory error: failed to allocate %ld bytes.", nbytes);
    return ptr;
}

void free_mem(void* ptr) noexcept {
    std::free(ptr);
}


namespace gten {

Tensor::Tensor(const std::vector<int32_t>& shape, Dtype dtype)
    : dtype_(dtype)
{
    refcount_ = reinterpret_cast<int32_t*>(alloc_mem(4));
    *refcount_ = 1;

    validate_shape(shape);
    shape_ = shape;
    set_strides(shape);
    numel_ = numel_from_shape();
    storage_size_ = nbytes();

    data_ = alloc_mem(nbytes());
}

Tensor::Tensor(const void *src_data, const std::vector<int32_t>& shape, Dtype dtype)
    : dtype_(dtype)
{
    refcount_ = reinterpret_cast<int32_t*>(alloc_mem(4));
    *refcount_ = 1;

    validate_shape(shape);
    shape_ = shape;
    set_strides(shape);
    numel_ = numel_from_shape();
    storage_size_ = nbytes();

    data_ = alloc_mem(nbytes());
    std::memcpy(data_, src_data, nbytes());
}

Tensor::Tensor(const Tensor& rhs) noexcept {
    if (this != &rhs)
    {
        // We are constructing a new tensor with data copied from an existing tensor.
        // So we just copy the data and refcount and incref it.
        shape_ = rhs.shape_;
        strides_ = rhs.strides_;
        dtype_ = rhs.dtype_;
        numel_ = rhs.numel_;
        storage_size_ = rhs.storage_size_;
        data_ = rhs.data_;
        refcount_ = rhs.refcount_;
        incref();
    }
}

Tensor& Tensor::operator=(const Tensor& rhs) noexcept {
    if (this != &rhs)
    {
        // We are copying data from an existing tensor to an existing tensor(this).
        // So we decref on the data we are holding, release it, replace it by copy
        // and incref the copied.
        shape_ = rhs.shape_;
        strides_ = rhs.strides_;
        dtype_ = rhs.dtype_;
        numel_ = rhs.numel_;
        storage_size_ = rhs.storage_size_;
        decref(); // Decref the current refcount.
        data_ = rhs.data_;  // Copy data.
        refcount_ = rhs.refcount_; // Replace by copied decref.
        incref(); // Incref the copied refcount.
    }
    return *this;
}

Tensor::Tensor(Tensor&& rhs) noexcept {
    if (this != &rhs)
    {
        // 'Steal' data from the other tensor. We must leave the other tensor in an invalid
        // state to prevent it to incref/decref the tensor data.
        shape_ = std::move(rhs.shape_);
        strides_ = std::move(rhs.strides_);
        dtype_ = rhs.dtype_;
        numel_ = rhs.numel_;
        storage_size_ = rhs.storage_size_;
        data_ = rhs.data_;
        refcount_ = rhs.refcount_;

        rhs.data_ = nullptr;
        rhs.refcount_ = nullptr;
    }
}

Tensor& Tensor::operator=(Tensor&& rhs) noexcept {
    if (this != &rhs)
    {
        // 'Steal' data from the other tensor. We must leave the other tensor in an invalid
        // state to prevent it to incref/decref the tensor data.
        shape_ = std::move(rhs.shape_);
        strides_ = std::move(rhs.strides_);
        dtype_ = rhs.dtype_;
        numel_ = rhs.numel_;
        storage_size_ = rhs.storage_size_;
        data_ = rhs.data_;
        refcount_ = rhs.refcount_;

        rhs.data_ = nullptr;
        rhs.refcount_ = nullptr;
    }
    return *this;
}

Tensor::~Tensor()
{
    decref();
}

void Tensor::incref() {
    if (refcount_)
        *refcount_ = *refcount_ + 1;
}

void Tensor::decref() {
    if (refcount_) {
        *refcount_ = *refcount_ - 1;
        if (*refcount_ == 0) {
            free_mem(data_);
            free_mem(refcount_);
            refcount_ = nullptr;
        }
    }
}

Dtype Tensor::dtype() const noexcept {
    return dtype_;
}

size_t Tensor::itemsize() const noexcept {
    if (dtype_ == kFloat16)
        return 2;
    return 4;
}

bool Tensor::is_1d() const noexcept {
    return shape_.size() == 1;
}

bool Tensor::is_2d() const noexcept {
    return shape_.size() == 2;
}

int32_t Tensor::ndims() const noexcept {
    return shape_.size();
}

int32_t Tensor::numel() const noexcept {
    return numel_;
}

void Tensor::resize(const std::vector<int32_t>& shape) noexcept {
    validate_shape(shape);
    shape_ = shape;
    set_strides(shape_);
    numel_ = numel_from_shape();
    GTEN_ASSERT(
        nbytes() <= storage_size_,
        "Resize size: %ld, exceeds preallocated size: %ld.",
        nbytes(),
        storage_size_);
}

void Tensor::resize(std::vector<int32_t> &&shape) noexcept
{
    validate_shape(shape);
    shape_ = std::move(shape);
    set_strides(shape_);
    numel_ = numel_from_shape();
    GTEN_ASSERT(
        nbytes() <= storage_size_,
        "Resize size: %ld, exceeds preallocated size: %ld.",
        nbytes(),
        storage_size_);
}

void Tensor::set_strides(const std::vector<int32_t> &shape)
{
    uint32_t n_dims = shape.size();
    if (n_dims == 1)
        strides_ = std::vector<int32_t>({1});
    else if (n_dims == 2)
        strides_ = std::vector<int32_t>({shape[1], 1});
    else if (n_dims == 3)
        strides_ = std::vector<int32_t>({shape[2] * shape[1], shape[1], 1});
}

int32_t Tensor::numel_from_shape() const noexcept {
    int32_t numel = 1;
    for (int32_t size : shape_)
        numel *= size;
    return numel;
}

int32_t Tensor::size(int32_t i) const
{
    GTEN_ASSERT(
        i < ndims(),
        "Tensor dim access, %d, is out of range of a tensor with %d-dims.",
        i,
        ndims());
    return shape_[i];
}

const std::vector<int32_t>& Tensor::shape() const noexcept
{
    return shape_;
}

uint64_t Tensor::nbytes() const noexcept
{
    return numel_ * itemsize();
}

void Tensor::print_single(int32_t item_idx, int32_t col_idx, int32_t n_cols) const noexcept
{
    uint32_t max_cols = dtype_ == kInt32 ? 32 : 8;
    if (dtype_ == kFloat16)
        std::cout << std::fixed
                  << std::setprecision(4)
                  << std::setw(7)
                  << fp16_to_fp32(reinterpret_cast<Float16*>(data_)[item_idx]);
    else if (dtype_ == kFloat32)
        std::cout << std::fixed
                  << std::setprecision(4)
                  << std::setw(7)
                  << reinterpret_cast<Float32*>(data_)[item_idx];
    else if (dtype_ ==kInt32)
        std::cout << reinterpret_cast<Int32*>(data_)[item_idx];
    if (col_idx != n_cols - 1)
        std::cout << ", ";
    if (col_idx > 0 && (col_idx % max_cols) == 0)
        std::cout << "\n  ";
}

void Tensor::print() const noexcept
{
    std::cout << "Tensor(\n";

    if (dtype_ == kFloat16)
        std::cout << "Numel=" << numel_ << "\nDtype=Float16\n[";
    else if (dtype_ == kFloat32)
        std::cout << "Numel=" << numel_ << "\nDtype=Float32\n[";
    else if (dtype_ == kInt32)
        std::cout << "Numel=" << numel_ << "\nDtype=Int32\n[";

    const uint32_t n_dims = shape_.size();
    if (n_dims == 1)
    {
        for (int col = 0; col < numel_; col += strides_[0])
            print_single(col * strides_[0], col, numel_);
    }
    else if (n_dims == 2)
    {
        const uint32_t n_rows = shape_[0];
        const uint32_t n_cols = shape_[1];
        for (int row = 0; row < n_rows; row++)
        {
            if (row == 0) std::cout << "[";
            else std::cout << " [";
            for (int col = 0; col < n_cols; col++)
            {
                const int idx = row * strides_[0] + col * strides_[1];
                if (idx >= numel_)
                    break;
                print_single(idx, col, n_cols);
            }
            if (row != n_rows - 1) std::cout << "]\n";
            else std::cout << "]";
        }
    }
    else // ndims=3
    {
        const int n_depth = shape_[0];
        const int n_rows = shape_[1];
        const int n_cols = shape_[2];
        for (int depth = 0; depth < n_depth; depth++)
        {
            if (depth == 0) std::cout << "[";
            else std::cout << " [";
            for (int row = 0; row < n_rows; row++)
            {
                if (row == 0) std::cout << "[";
                else std::cout << "  [";
                for (int col = 0; col < n_cols; col++)
                {
                    const int idx = (depth * strides_[0]) + (row * strides_[1]) + col* strides_[2];
                    if (idx >= numel_)
                        break;
                    print_single(idx, col, n_cols);
                }
                std::cout << "]";
                if (row != n_rows - 1)
                    std::cout << "\n";
            }
            std::cout << "]";
            if (depth != n_depth - 1)
                std::cout << "\n\n";
        }
        
    }
    std::cout << "])\n";
}

void Tensor::validate_shape(const std::vector<int32_t> &shape) const
{
    if (shape.size() == 0)
        GTEN_ASSERT(false, "Creation of tensor with no shape is not allowed");
    if (shape.size() > 3)
        GTEN_ASSERT(false, "Creation of tensors with dims > 4 is not supported.");
    for (uint32_t size : shape_)
        if (size == 0)
            GTEN_ASSERT(false, "One of the provided dimensions in the shape is zero.");
}

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor) {
    tensor.print();
    return stream;
}

} // namespace gten
