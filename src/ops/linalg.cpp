#include "kael/factory.h"
#include "kael/ops.h"

namespace kael {

namespace {

template <typename T>
void multiply_matrices(const T* a, const T* b, T* out, std::int64_t rows, std::int64_t inner,
                       std::int64_t columns) {
    for (std::int64_t i = 0; i < rows; ++i) {
        T* out_row = out + i * columns;
        for (std::int64_t k = 0; k < inner; ++k) {
            const T a_ik = a[i * inner + k];
            const T* b_row = b + k * columns;
            for (std::int64_t j = 0; j < columns; ++j) {
                out_row[j] = static_cast<T>(out_row[j] + a_ik * b_row[j]);
            }
        }
    }
}

struct MatmulLayout {
    Shape batch;
    std::int64_t rows;
    std::int64_t inner;
    std::int64_t columns;
};

Shape batch_shape_of(const Tensor& matrix) {
    return Shape(matrix.shape().begin(), matrix.shape().end() - 2);
}

Shape with_matrix_dims(Shape batch, std::int64_t rows, std::int64_t columns) {
    batch.push_back(rows);
    batch.push_back(columns);
    return batch;
}

Tensor as_batch_of_matrices(const Tensor& matrix, const Shape& batch, std::int64_t rows,
                            std::int64_t columns) {
    return matrix.expand(with_matrix_dims(batch, rows, columns))
        .reshape({numel(batch), rows, columns});
}

Tensor multiply_batches(const Tensor& a, const Tensor& b, const MatmulLayout& layout) {
    const DType dtype = promote_types(a.dtype(), b.dtype());
    KAEL_CHECK(dtype != DType::Bool, "matmul does not support dtype bool");

    const auto [batch, rows, inner, columns] = layout;
    const Tensor flat_a = as_batch_of_matrices(a.to(dtype), batch, rows, inner);
    const Tensor flat_b = as_batch_of_matrices(b.to(dtype), batch, inner, columns);
    Tensor out = zeros(with_matrix_dims(batch, rows, columns), dtype);

    const std::int64_t batch_count = numel(batch);
    dispatch_no_bool(dtype, "matmul", [&]<typename T>() {
        const T* data_a = flat_a.data_ptr<T>();
        const T* data_b = flat_b.data_ptr<T>();
        T* data_out = out.data_ptr<T>();
        for (std::int64_t i = 0; i < batch_count; ++i) {
            multiply_matrices(data_a + i * rows * inner, data_b + i * inner * columns,
                              data_out + i * rows * columns, rows, inner, columns);
        }
    });
    return out;
}

} // namespace

Tensor matmul(const Tensor& a, const Tensor& b) {
    KAEL_CHECK(a.dim() >= 1 && b.dim() >= 1, "matmul needs at least 1 dimension, got shapes "
                                                 << to_string(a.shape()) << " and "
                                                 << to_string(b.shape()));

    const bool a_is_vector = a.dim() == 1;
    const bool b_is_vector = b.dim() == 1;
    const Tensor left = a_is_vector ? a.unsqueeze(0) : a;
    const Tensor right = b_is_vector ? b.unsqueeze(1) : b;

    KAEL_CHECK(left.size(-1) == right.size(-2), "matmul shapes " << to_string(a.shape()) << " and "
                                                                 << to_string(b.shape())
                                                                 << " do not align");

    const MatmulLayout layout{broadcast_shapes(batch_shape_of(left), batch_shape_of(right)),
                              left.size(-2), left.size(-1), right.size(-1)};
    Tensor out = multiply_batches(left, right, layout);
    if (b_is_vector) {
        out = out.squeeze(-1);
    }
    return a_is_vector ? out.squeeze(b_is_vector ? -1 : -2) : out;
}

Tensor dot(const Tensor& a, const Tensor& b) {
    KAEL_CHECK(a.dim() == 1 && b.dim() == 1, "dot needs 1-d tensors, got shapes "
                                                 << to_string(a.shape()) << " and "
                                                 << to_string(b.shape()));
    return matmul(a, b);
}

Tensor outer(const Tensor& a, const Tensor& b) {
    KAEL_CHECK(a.dim() == 1 && b.dim() == 1, "outer needs 1-d tensors, got shapes "
                                                 << to_string(a.shape()) << " and "
                                                 << to_string(b.shape()));
    return a.unsqueeze(1) * b.unsqueeze(0);
}

Tensor mm(const Tensor& a, const Tensor& b) {
    KAEL_CHECK(a.dim() == 2 && b.dim() == 2, "mm needs 2-d tensors, got shapes "
                                                 << to_string(a.shape()) << " and "
                                                 << to_string(b.shape()));
    return matmul(a, b);
}

Tensor bmm(const Tensor& a, const Tensor& b) {
    KAEL_CHECK(a.dim() == 3 && b.dim() == 3, "bmm needs 3-d tensors, got shapes "
                                                 << to_string(a.shape()) << " and "
                                                 << to_string(b.shape()));
    return matmul(a, b);
}

Tensor Tensor::matmul(const Tensor& other) const {
    return kael::matmul(*this, other);
}

} // namespace kael
