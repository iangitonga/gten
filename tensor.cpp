#include "tensor.h"

#include <iostream>
#include <stdexcept>
#include <cstring>



namespace gten {

Tensor::Tensor(const std::vector<uint32_t> &shape, Dtype dtype)
    : m_dtype(dtype)
{
    validate_shape(shape);
    m_shape = shape;
    set_strides(shape);
    m_numel = numel_from_shape();
    if (dtype == Dtype::Float32)
        m_data = std::shared_ptr<float>(new float[m_numel]);
    else if (dtype == Dtype::Int32)
        m_data = std::shared_ptr<int>(new int[m_numel]);
}

Tensor::Tensor(const void *src_data, const std::vector<uint32_t> &shape, Dtype dtype)
    : m_dtype(dtype)
{
    validate_shape(shape);
    m_shape = shape;
    set_strides(shape);
    m_numel = numel_from_shape();

    if (dtype == Dtype::Float32) {
        m_data = std::shared_ptr<float>(new float[m_numel]);
        std::memcpy(m_data.get(), src_data, m_numel * 4);
    } else if (dtype == Dtype::Int32) {
        m_data = std::shared_ptr<int>(new int[m_numel]);
        std::memcpy(m_data.get(), src_data, m_numel * 4);
    }
}


void Tensor::set_strides(const std::vector<uint32_t> &shape)
{
    uint32_t n_dims = shape.size();
    if (n_dims == 1)
        m_strides = std::vector<uint32_t>({1});
    else if (n_dims == 2)
        m_strides = std::vector<uint32_t>({shape[1], 1});
    else if (n_dims == 3)
        m_strides = std::vector<uint32_t>({shape[2] * shape[1], shape[1], 1});
}

uint32_t Tensor::numel_from_shape() {
    uint32_t numel = 1;
    for (uint32_t size : m_shape)
        numel *= size;
    return numel;
}


void Tensor::copy_from_other(const Tensor &other) {
    if (this != &other)
    {
        m_shape = other.m_shape;
        m_strides = other.m_strides;
        m_dtype = other.m_dtype;
        m_numel = other.m_numel;
        m_data = other.m_data;
    }
}

void Tensor::move_from_other(Tensor &&other)
{
    if (this != &other)
    {
        m_shape = std::move(other.m_shape);
        m_strides = std::move(other.m_strides);
        m_dtype = other.m_dtype;
        m_numel = other.m_numel;
        m_data = std::move(other.m_data);
    }
}

Tensor Tensor::to(const Dtype dtype)
{
    // TODO: Should we allocate a new tensor if the types are the same.
    if (m_dtype == dtype)
        return *this;

    Tensor out_tensor(m_shape, dtype);

    if (m_dtype == Dtype::Float32)
    {
        // float to int
        float *src_data = data_ptr<float>();
        uint32_t *dest_data = out_tensor.data_ptr<uint32_t>();
        for (uint32_t i = 0; i < m_numel; i++)
            dest_data[i] = static_cast<int32_t>(src_data[i]);
    }
    else
    {
        // int to float
        uint32_t *src_data = data_ptr<uint32_t>();
        float *dest_data = out_tensor.data_ptr<float>();
        for (uint32_t i = 0; i < m_numel; i++)
            dest_data[i] = static_cast<float>(src_data[i]);

    }

    return out_tensor;
}

void Tensor::print_single(const uint32_t item_idx, const uint32_t col_idx, const uint32_t n_cols) const
{
    uint32_t max_cols = m_dtype == Dtype::Float32 ? 8 : 32;
    if (m_dtype == Dtype::Float32)
        std::cout << ((float *)m_data.get())[item_idx];
    else if (m_dtype == Dtype::Int32)
        std::cout << ((int32_t *)m_data.get())[item_idx];
    if (col_idx != n_cols - 1)
        std::cout << ", ";
    if (col_idx > 0 && (col_idx % max_cols) == 0)
        std::cout << "\n  ";
}

void Tensor::print() const
{
    std::cout << "Tensor(\n";

    if (m_dtype == Dtype::Float32)
        std::cout << "Numel=" << m_numel << "\nDtype=Float32\n[";
    else if (m_dtype == Dtype::Int32)
        std::cout << "Numel=" << m_numel << "\nDtype=Int32\n[";

    const uint32_t n_dims = m_shape.size();
    if (n_dims == 1)
    {
        for (int col = 0; col < m_numel; col += m_strides[0])
            print_single(col * m_strides[0], col, m_numel);
    }
    else if (n_dims == 2)
    {
        const uint32_t n_rows = m_shape[0];
        const uint32_t n_cols = m_shape[1];
        for (int row = 0; row < n_rows; row++)
        {
            if (row == 0) std::cout << "[";
            else std::cout << " [";
            for (int col = 0; col < n_cols; col++)
            {
                const int idx = row * m_strides[0] + col * m_strides[1];
                if (idx >= m_numel)
                    break;
                print_single(idx, col, n_cols);
            }
            if (row != n_rows - 1) std::cout << "]\n";
            else std::cout << "]";
        }
    }
    else // ndims=3
    {
        const uint32_t n_depth = m_shape[0];
        const uint32_t n_rows = m_shape[1];
        const uint32_t n_cols = m_shape[2];
        for (uint32_t depth = 0; depth < n_depth; depth++)
        {
            if (depth == 0) std::cout << "[";
            else std::cout << " [";
            for (uint32_t row = 0; row < n_rows; row++)
            {
                if (row == 0) std::cout << "[";
                else std::cout << "  [";
                for (int col = 0; col < n_cols; col++)
                {
                    const int idx = (depth * m_strides[0]) + (row * m_strides[1]) + col* m_strides[2];
                    if (idx >= m_numel)
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

void Tensor::validate_shape(const std::vector<uint32_t> &shape) const
{
    if (shape.size() == 0)
        throw std::runtime_error("Creation of tensor with no shape is not allowed");
    if (shape.size() > 3)
        throw std::runtime_error("Creation of tensors with dims > 4 is not supported.");
    for (uint32_t size : m_shape)
        if (size == 0)
            throw std::runtime_error("One of the provided dimensions in the shape is zero.");
}

} // namespace gten

