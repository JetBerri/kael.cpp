#include "kael/autograd/functions.h"
#include "kael/ops.h"

#include "record.h"
#include "recording.h"

namespace kael::autograd {

namespace {

struct MatmulLayout {
    bool left_is_vector;
    bool right_is_vector;

    [[nodiscard]] Tensor matrix_gradient(const Tensor& g) const {
        const Tensor with_columns = right_is_vector ? g.unsqueeze(-1) : g;
        return left_is_vector ? with_columns.unsqueeze(-2) : with_columns;
    }
    [[nodiscard]] Tensor left_matrix(const Tensor& a) const {
        return left_is_vector ? a.unsqueeze(0) : a;
    }
    [[nodiscard]] Tensor right_matrix(const Tensor& b) const {
        return right_is_vector ? b.unsqueeze(1) : b;
    }
};

} // namespace

void record_matmul(const Tensor& out, const Tensor& a, const Tensor& b) {
    if (!should_record(out, {&a, &b})) {
        return;
    }
    const MatmulLayout layout{a.dim() == 1, b.dim() == 1};
    set_history(out, "MatmulBackward", {a, b},
                {[layout, right = SavedTensor(layout.right_matrix(b)), shape = a.shape(),
                  dtype = a.dtype()](const Tensor& g) {
                     const Tensor gradient = matmul(layout.matrix_gradient(g), right.get().mT());
                     const Tensor reduced =
                         sum_to_shape(gradient, layout.left_is_vector ? Shape{1, shape[0]} : shape);
                     return (layout.left_is_vector ? reduced.squeeze(0) : reduced).to(dtype);
                 },
                 [layout, left = SavedTensor(layout.left_matrix(a)), shape = b.shape(),
                  dtype = b.dtype()](const Tensor& g) {
                     const Tensor gradient = matmul(left.get().mT(), layout.matrix_gradient(g));
                     const Tensor reduced = sum_to_shape(
                         gradient, layout.right_is_vector ? Shape{shape[0], 1} : shape);
                     return (layout.right_is_vector ? reduced.squeeze(-1) : reduced).to(dtype);
                 }});
}

} // namespace kael::autograd
