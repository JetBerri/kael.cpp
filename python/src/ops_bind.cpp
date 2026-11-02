#include <algorithm>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "bind.h"

namespace kael::python {

namespace {

using UnaryFunction = Tensor (*)(const Tensor&);

template <UnaryFunction F>
void add_unary(nb::module_& m, nb::class_<Tensor>& tensor, const char* name, const char* doc) {
    m.def(name, F, nb::arg("input"), doc, nb::call_guard<nb::gil_scoped_release>());
    tensor.def(name, F, doc, nb::call_guard<nb::gil_scoped_release>());
}

template <typename Function>
void add_reduction(nb::module_& m, nb::class_<Tensor>& tensor, const char* name, Function function,
                   const char* doc) {
    const auto wrapped = [function](const Tensor& t, const nb::object& dim, bool keepdim) {
        const Dims dims = dims_from(dim);
        const nb::gil_scoped_release release;
        return function(t, dims, keepdim);
    };
    m.def(name, wrapped, nb::arg("input"), nb::arg("dim") = nb::none(), nb::arg("keepdim") = false,
          doc);
    tensor.def(name, wrapped, nb::arg("dim") = nb::none(), nb::arg("keepdim") = false, doc);
}

template <Tensor (*F)(const Operand&, const Operand&)>
void add_binary(nb::module_& m, const char* name, const char* doc) {
    m.def(
        name, [](nb::handle a, nb::handle b) { return F(operand_from(a), operand_from(b)); },
        nb::arg("a"), nb::arg("b"), doc);
}

} // namespace

void bind_ops(nb::module_& m) {
    auto tensor = nb::borrow<nb::class_<Tensor>>(m.attr("Tensor"));

    add_unary<&kael::neg>(m, tensor, "neg", "Elementwise negation.");
    add_unary<&kael::abs>(m, tensor, "abs", "Elementwise absolute value.");
    add_unary<&kael::sign>(m, tensor, "sign", "Elementwise sign.");
    add_unary<&kael::exp>(m, tensor, "exp", "Elementwise exponential.");
    add_unary<&kael::log>(m, tensor, "log", "Elementwise natural logarithm.");
    add_unary<&kael::log1p>(m, tensor, "log1p", "Elementwise log(1 + x).");
    add_unary<&kael::expm1>(m, tensor, "expm1", "Elementwise exp(x) - 1.");
    add_unary<&kael::sqrt>(m, tensor, "sqrt", "Elementwise square root.");
    add_unary<&kael::rsqrt>(m, tensor, "rsqrt", "Elementwise reciprocal square root.");
    add_unary<&kael::sin>(m, tensor, "sin", "Elementwise sine.");
    add_unary<&kael::cos>(m, tensor, "cos", "Elementwise cosine.");
    add_unary<&kael::tanh>(m, tensor, "tanh", "Elementwise hyperbolic tangent.");
    add_unary<&kael::sigmoid>(m, tensor, "sigmoid", "Elementwise logistic function.");
    add_unary<&kael::relu>(m, tensor, "relu", "Elementwise rectified linear unit.");
    add_unary<&kael::logical_not>(m, tensor, "logical_not", "Elementwise logical not.");

    const auto gelu = [](const Tensor& t, bool approximate) { return kael::gelu(t, approximate); };
    m.def("gelu", gelu, nb::arg("input"), nb::arg("approximate") = false,
          "Gaussian error linear unit.");
    tensor.def("gelu", gelu, nb::arg("approximate") = false);

    add_binary<&kael::add>(m, "add", "Elementwise a + b with broadcasting.");
    add_binary<&kael::sub>(m, "sub", "Elementwise a - b with broadcasting.");
    add_binary<&kael::mul>(m, "mul", "Elementwise a * b with broadcasting.");
    add_binary<&kael::div>(m, "div", "Elementwise a / b with broadcasting.");
    add_binary<&kael::floor_divide>(m, "floor_divide", "Elementwise a // b.");
    add_binary<&kael::pow>(m, "pow", "Elementwise a ** b.");
    add_binary<&kael::maximum>(m, "maximum", "Elementwise maximum, propagating NaN.");
    add_binary<&kael::minimum>(m, "minimum", "Elementwise minimum, propagating NaN.");
    add_binary<&kael::eq>(m, "eq", "Elementwise a == b.");
    add_binary<&kael::ne>(m, "ne", "Elementwise a != b.");
    add_binary<&kael::lt>(m, "lt", "Elementwise a < b.");
    add_binary<&kael::le>(m, "le", "Elementwise a <= b.");
    add_binary<&kael::gt>(m, "gt", "Elementwise a > b.");
    add_binary<&kael::ge>(m, "ge", "Elementwise a >= b.");
    add_binary<&kael::logical_and>(m, "logical_and", "Elementwise logical and.");
    add_binary<&kael::logical_or>(m, "logical_or", "Elementwise logical or.");
    add_binary<&kael::logical_xor>(m, "logical_xor", "Elementwise logical xor.");

    add_reduction(
        m, tensor, "sum", [](const Tensor& t, const Dims& d, bool k) { return kael::sum(t, d, k); },
        "Sum over the given dimensions.");
    add_reduction(
        m, tensor, "prod",
        [](const Tensor& t, const Dims& d, bool k) { return kael::prod(t, d, k); },
        "Product over the given dimensions.");
    add_reduction(
        m, tensor, "mean",
        [](const Tensor& t, const Dims& d, bool k) { return kael::mean(t, d, k); },
        "Mean over the given dimensions.");
    add_reduction(
        m, tensor, "max", [](const Tensor& t, const Dims& d, bool k) { return kael::max(t, d, k); },
        "Maximum over the given dimensions.");
    add_reduction(
        m, tensor, "min", [](const Tensor& t, const Dims& d, bool k) { return kael::min(t, d, k); },
        "Minimum over the given dimensions.");
    add_reduction(
        m, tensor, "norm",
        [](const Tensor& t, const Dims& d, bool k) { return kael::norm(t, d, k); },
        "Euclidean norm over the given dimensions.");
    add_reduction(
        m, tensor, "all", [](const Tensor& t, const Dims& d, bool k) { return kael::all(t, d, k); },
        "Whether all elements are true.");
    add_reduction(
        m, tensor, "any", [](const Tensor& t, const Dims& d, bool k) { return kael::any(t, d, k); },
        "Whether any element is true.");

    const auto var = [](const Tensor& t, const nb::object& dim, bool keepdim,
                        std::int64_t correction) {
        return kael::var(t, dims_from(dim), keepdim, correction);
    };
    const auto std_dev = [](const Tensor& t, const nb::object& dim, bool keepdim,
                            std::int64_t correction) {
        return kael::std(t, dims_from(dim), keepdim, correction);
    };
    m.def("var", var, nb::arg("input"), nb::arg("dim") = nb::none(), nb::arg("keepdim") = false,
          nb::arg("correction") = 1);
    tensor.def("var", var, nb::arg("dim") = nb::none(), nb::arg("keepdim") = false,
               nb::arg("correction") = 1);
    m.def("std", std_dev, nb::arg("input"), nb::arg("dim") = nb::none(), nb::arg("keepdim") = false,
          nb::arg("correction") = 1);
    tensor.def("std", std_dev, nb::arg("dim") = nb::none(), nb::arg("keepdim") = false,
               nb::arg("correction") = 1);

    const auto argmax = [](const Tensor& t, const std::optional<std::int64_t>& dim, bool keepdim) {
        return dim ? kael::argmax(t, *dim, keepdim) : kael::argmax(t);
    };
    const auto argmin = [](const Tensor& t, const std::optional<std::int64_t>& dim, bool keepdim) {
        return dim ? kael::argmin(t, *dim, keepdim) : kael::argmin(t);
    };
    m.def("argmax", argmax, nb::arg("input"), nb::arg("dim") = nb::none(),
          nb::arg("keepdim") = false);
    tensor.def("argmax", argmax, nb::arg("dim") = nb::none(), nb::arg("keepdim") = false);
    m.def("argmin", argmin, nb::arg("input"), nb::arg("dim") = nb::none(),
          nb::arg("keepdim") = false);
    tensor.def("argmin", argmin, nb::arg("dim") = nb::none(), nb::arg("keepdim") = false);

    const auto matmul = [](const Tensor& a, const Tensor& b) { return kael::matmul(a, b); };
    m.def("matmul", matmul, nb::call_guard<nb::gil_scoped_release>(),
          "Matrix product with batches.");
    tensor.def("matmul", matmul, nb::call_guard<nb::gil_scoped_release>());
    m.def("dot", &kael::dot);
    m.def("outer", &kael::outer);
    m.def("mm", &kael::mm);
    m.def("bmm", &kael::bmm);

    m.def("cat", &kael::cat, nb::arg("tensors"), nb::arg("dim") = 0,
          "Join tensors along a dimension.");
    m.def("stack", &kael::stack, nb::arg("tensors"), nb::arg("dim") = 0,
          "Join tensors along a new dimension.");
    m.def(
        "where",
        [](const Tensor& condition, nb::handle a, nb::handle b) {
            return kael::where(condition, operand_from(a), operand_from(b));
        },
        "Pick from a where the condition holds and from b elsewhere.");
    m.def("masked_fill", [](const Tensor& t, const Tensor& mask, nb::handle value) {
        return kael::masked_fill(t, mask, scalar_from(value));
    });
    tensor.def("masked_fill", [](const Tensor& t, const Tensor& mask, nb::handle value) {
        return kael::masked_fill(t, mask, scalar_from(value));
    });
    m.def("index_select", &kael::index_select);
    tensor.def("index_select", [](const Tensor& t, std::int64_t dim, const Tensor& indices) {
        return kael::index_select(t, dim, indices);
    });
    m.def("gather", &kael::gather);
    tensor.def("gather", [](const Tensor& t, std::int64_t dim, const Tensor& indices) {
        return kael::gather(t, dim, indices);
    });
    tensor.def("scatter_add_",
               [](nb::handle self, std::int64_t dim, const Tensor& indices, const Tensor& source) {
                   nb::cast<Tensor&>(self).scatter_add_(dim, indices, source);
                   return nb::borrow(self);
               });
    tensor.def("index_add_",
               [](nb::handle self, std::int64_t dim, const Tensor& indices, const Tensor& source) {
                   nb::cast<Tensor&>(self).index_add_(dim, indices, source);
                   return nb::borrow(self);
               });

    m.def("logsumexp", &kael::logsumexp, nb::arg("input"), nb::arg("dim"),
          nb::arg("keepdim") = false);
    const auto softmax = [](const Tensor& t, std::int64_t dim) { return kael::softmax(t, dim); };
    const auto log_softmax = [](const Tensor& t, std::int64_t dim) {
        return kael::log_softmax(t, dim);
    };
    m.def("softmax", softmax, nb::arg("input"), nb::arg("dim"));
    tensor.def("softmax", softmax, nb::arg("dim"));
    m.def("log_softmax", log_softmax, nb::arg("input"), nb::arg("dim"));
    m.def("cross_entropy_losses", &kael::cross_entropy_losses, nb::arg("logits"), nb::arg("targets"));
    tensor.def("log_softmax", log_softmax, nb::arg("dim"));
    m.def("standardize", &kael::standardize);
    m.def("clamp", [](const Tensor& t, nb::handle low, nb::handle high) {
        return kael::clamp(t, scalar_from(low), scalar_from(high));
    });
    tensor.def("clamp", [](const Tensor& t, nb::handle low, nb::handle high) {
        return kael::clamp(t, scalar_from(low), scalar_from(high));
    });

    m.def("empty", [](const Shape& shape, DType dtype, const std::string& device) {
        return kael::empty(shape, dtype, parse_device(device));
    });
    m.def("zeros", [](const Shape& shape, DType dtype, const std::string& device) {
        return kael::zeros(shape, dtype, parse_device(device));
    });
    m.def("ones", [](const Shape& shape, DType dtype, const std::string& device) {
        return kael::ones(shape, dtype, parse_device(device));
    });
    m.def("full", [](const Shape& shape, nb::handle value, DType dtype, const std::string& device) {
        return kael::full(shape, scalar_from(value), dtype, parse_device(device));
    });
    m.def("zeros_like", &kael::zeros_like);
    m.def("ones_like", &kael::ones_like);
    m.def("empty_like", &kael::empty_like);
    m.def("arange",
          [](double start, double end, double step, DType dtype, const std::string& device) {
              return kael::arange(start, end, step, dtype).to(parse_device(device));
          });
    m.def("linspace",
          [](double start, double end, std::int64_t steps, DType dtype, const std::string& device) {
              return kael::linspace(start, end, steps, dtype).to(parse_device(device));
          });
    m.def("eye", [](std::int64_t n, DType dtype, const std::string& device) {
        return kael::eye(n, dtype).to(parse_device(device));
    });
    m.def("rand", [](const Shape& shape, DType dtype, const std::string& device) {
        return kael::rand(shape, dtype).to(parse_device(device));
    });
    m.def("randn", [](const Shape& shape, DType dtype, const std::string& device) {
        return kael::randn(shape, dtype).to(parse_device(device));
    });
    m.def("randint",
          [](std::int64_t low, std::int64_t high, const Shape& shape, const std::string& device) {
              return kael::randint(low, high, shape).to(parse_device(device));
          });
    m.def("randperm", [](std::int64_t n, const std::string& device) {
        return kael::randperm(n).to(parse_device(device));
    });
    m.def("bernoulli",
          [](const Shape& shape, double probability, DType dtype, const std::string& device) {
              return kael::bernoulli(shape, probability, dtype).to(parse_device(device));
          });
    m.def("manual_seed", &kael::manual_seed, "Seed the global random generator.");

    m.def("set_num_threads", &kael::set_num_threads);
    m.def("get_num_threads", &kael::get_num_threads);
    m.def("cuda_is_available", &kael::cuda_is_available);
    m.def("cuda_synchronize", &kael::cuda_synchronize);
    m.def("empty_cache", &kael::empty_cache);
    m.def(
        "allocator_stats",
        [](const std::string& device) {
            const AllocatorStats stats = kael::allocator_stats(parse_device(device));
            nb::dict result;
            result["reserved_bytes"] = stats.reserved_bytes;
            result["in_use_bytes"] = stats.in_use_bytes;
            result["cached_bytes"] = stats.cached_bytes;
            result["hits"] = stats.hits;
            result["misses"] = stats.misses;
            return result;
        },
        nb::arg("device") = "cpu");
    m.def("equal", &kael::equal, "Whether two tensors have the same shape and values.");
}

} // namespace kael::python
