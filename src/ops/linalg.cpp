#include <algorithm>
#include <type_traits>

#include "kael/autograd/grad_mode.h"
#include "kael/factory.h"
#include "kael/matmul_versions.h"
#include "kael/ops.h"
#include "kael/parallel.h"

#include "autograd/record.h"

namespace kael {

namespace {

constexpr std::int64_t kWorkPerParallelTask = std::int64_t{1} << 21;

template <typename T>
void multiply_exact(const T* a, const T* b, T* out, const matmul_versions::Sizes& sizes) {
    const auto [rows, inner, columns] = sizes;
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

template <typename T>
void multiply_matrices(const T* a, const T* b, T* out, const matmul_versions::Sizes& sizes) {
    if constexpr (std::is_floating_point_v<T>) {
        matmul_versions::multiply(a, b, out, sizes);
    } else {
        multiply_exact(a, b, out, sizes);
    }
}

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

Tensor multiply_batches(const Tensor& a, const Tensor& b, const Shape& batch,
                        const matmul_versions::Sizes& sizes) {
    const DType dtype = promote_types(a.dtype(), b.dtype());
    const auto [rows, inner, columns] = sizes;
    const Tensor flat_a = as_batch_of_matrices(a.to(dtype), batch, rows, inner);
    const Tensor flat_b = as_batch_of_matrices(b.to(dtype), batch, inner, columns);
    Tensor out = zeros(with_matrix_dims(batch, rows, columns), dtype);

    const std::int64_t batch_count = numel(batch);
    dispatch_no_bool(dtype, "matmul", [&]<typename T>() {
        const T* data_a = flat_a.data_ptr<T>();
        const T* data_b = flat_b.data_ptr<T>();
        T* data_out = out.data_ptr<T>();
        const std::int64_t work = std::max<std::int64_t>(1, rows * inner * columns);
        parallel_for(0, batch_count, std::max<std::int64_t>(1, kWorkPerParallelTask / work),
                     [&](std::int64_t first, std::int64_t last) {
                         for (std::int64_t i = first; i < last; ++i) {
                             multiply_matrices(data_a + i * rows * inner,
                                               data_b + i * inner * columns,
                                               data_out + i * rows * columns, sizes);
                         }
                     });
    });
    return out;
}

void check_ranks(const Tensor& a, const Tensor& b, std::int64_t rank_a, std::int64_t rank_b,
                 const char* op) {
    KAEL_CHECK_VALUE(a.dim() == rank_a && b.dim() == rank_b,
                     op << " needs tensors with " << rank_a << " and " << rank_b
                        << " dimensions, got shapes " << to_string(a.shape()) << " and "
                        << to_string(b.shape()));
}

Tensor matmul_forward(const Tensor& a, const Tensor& b) {
    KAEL_CHECK_VALUE(a.dim() >= 1 && b.dim() >= 1, "matmul needs at least 1 dimension, got shapes "
                                                       << to_string(a.shape()) << " and "
                                                       << to_string(b.shape()));

    const bool a_is_vector = a.dim() == 1;
    const bool b_is_vector = b.dim() == 1;
    const Tensor left = a_is_vector ? a.unsqueeze(0) : a;
    const Tensor right = b_is_vector ? b.unsqueeze(1) : b;

    KAEL_CHECK_VALUE(left.size(-1) == right.size(-2),
                     "matmul shapes " << to_string(a.shape()) << " and " << to_string(b.shape())
                                      << " do not align");

    const Shape batch = broadcast_shapes(batch_shape_of(left), batch_shape_of(right));
    Tensor out =
        multiply_batches(left, right, batch, {left.size(-2), left.size(-1), right.size(-1)});
    if (b_is_vector) {
        out = out.squeeze(-1);
    }
    return a_is_vector ? out.squeeze(b_is_vector ? -1 : -2) : out;
}

} // namespace

Tensor matmul(const Tensor& a, const Tensor& b) {
    Tensor out = autograd::without_grad([&] { return matmul_forward(a, b); });
    autograd::record_matmul(out, a, b);
    return out;
}

Tensor dot(const Tensor& a, const Tensor& b) {
    check_ranks(a, b, 1, 1, "dot");
    return matmul(a, b);
}

Tensor outer(const Tensor& a, const Tensor& b) {
    check_ranks(a, b, 1, 1, "outer");
    return a.unsqueeze(1) * b.unsqueeze(0);
}

Tensor mm(const Tensor& a, const Tensor& b) {
    check_ranks(a, b, 2, 2, "mm");
    return matmul(a, b);
}

Tensor bmm(const Tensor& a, const Tensor& b) {
    check_ranks(a, b, 3, 3, "bmm");
    return matmul(a, b);
}

Tensor Tensor::matmul(const Tensor& other) const {
    return kael::matmul(*this, other);
}

} // namespace kael
