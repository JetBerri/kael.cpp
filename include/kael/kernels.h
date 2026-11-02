#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "kael/dtype.h"
#include "kael/iterator.h"
#include "kael/tensor.h"

namespace kael {

namespace detail {

template <typename Out, typename In, std::size_t... I, typename F>
void typed_inner_loop(std::byte* const* pointers, const std::int64_t* strides, std::int64_t count,
                      F& f, std::index_sequence<I...>) {
    const bool contiguous = strides[0] == static_cast<std::int64_t>(sizeof(Out)) &&
                            ((strides[I + 1] == static_cast<std::int64_t>(sizeof(In))) && ...);
    if (contiguous) {
        Out* out = reinterpret_cast<Out*>(pointers[0]);
        const In* const inputs[] = {reinterpret_cast<const In*>(pointers[I + 1])...};
        for (std::int64_t i = 0; i < count; ++i) {
            out[i] = convert_value<Out>(f(inputs[I][i]...));
        }
        return;
    }
    for (std::int64_t i = 0; i < count; ++i) {
        *reinterpret_cast<Out*>(pointers[0] + i * strides[0]) = convert_value<Out>(
            f(*reinterpret_cast<const In*>(pointers[I + 1] + i * strides[I + 1])...));
    }
}

template <typename Out, typename In, std::size_t InputCount, typename F>
void run_typed(const TensorIterator& iterator, F& f) {
    iterator.run([&](std::byte* const* pointers, const std::int64_t* strides, std::int64_t count) {
        typed_inner_loop<Out, In>(pointers, strides, count, f,
                                  std::make_index_sequence<InputCount>{});
    });
}

template <typename F, typename... Inputs>
Tensor same_dtype_kernel(DType dtype, F& f, const Inputs&... inputs) {
    TensorIterator iterator;
    iterator.allocate_output(dtype);
    (iterator.add_input(inputs.to(dtype)), ...);
    iterator.build();
    dispatch(dtype, [&]<typename T>() { run_typed<T, T, sizeof...(Inputs)>(iterator, f); });
    return iterator.output();
}

template <typename F, typename... Inputs>
void same_dtype_kernel_out(Tensor& out, F& f, const Inputs&... inputs) {
    TensorIterator iterator;
    iterator.add_output(out);
    (iterator.add_input(inputs.to(out.dtype())), ...);
    iterator.build();
    dispatch(out.dtype(), [&]<typename T>() { run_typed<T, T, sizeof...(Inputs)>(iterator, f); });
}

} // namespace detail

template <typename F>
Tensor unary_kernel(const Tensor& input, F f) {
    return detail::same_dtype_kernel(input.dtype(), f, input);
}

template <typename F>
Tensor floating_unary_kernel(const Tensor& input, const char* op, F f) {
    const DType dtype = is_floating_point(input.dtype()) ? input.dtype() : DType::F32;
    TensorIterator iterator;
    iterator.allocate_output(dtype).add_input(input.to(dtype)).build();
    dispatch_floating(dtype, op, [&]<typename T>() { detail::run_typed<T, T, 1>(iterator, f); });
    return iterator.output();
}

template <typename F>
void unary_kernel_out(Tensor& out, const Tensor& input, F f) {
    detail::same_dtype_kernel_out(out, f, input);
}

template <typename F>
Tensor binary_kernel(const Tensor& a, const Tensor& b, F f) {
    return detail::same_dtype_kernel(promote_types(a.dtype(), b.dtype()), f, a, b);
}

template <typename F>
void binary_kernel_out(Tensor& out, const Tensor& a, const Tensor& b, F f) {
    detail::same_dtype_kernel_out(out, f, a, b);
}

template <typename F>
Tensor ternary_kernel(const Tensor& a, const Tensor& b, const Tensor& c, F f) {
    const DType dtype = promote_types(promote_types(a.dtype(), b.dtype()), c.dtype());
    return detail::same_dtype_kernel(dtype, f, a, b, c);
}

template <typename F>
Tensor comparison_kernel(const Tensor& a, const Tensor& b, F f) {
    const DType dtype = promote_types(a.dtype(), b.dtype());
    TensorIterator iterator;
    iterator.allocate_output(DType::Bool).add_input(a.to(dtype)).add_input(b.to(dtype)).build();
    dispatch(dtype, [&]<typename T>() { detail::run_typed<std::uint8_t, T, 2>(iterator, f); });
    return iterator.output();
}

} // namespace kael
