#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <sstream>

#include "bind.h"

namespace kael::python {

namespace {

nb::tuple shape_tuple(const Shape& shape) {
    nb::list items;
    for (const std::int64_t size : shape) {
        items.append(size);
    }
    return nb::tuple(items);
}

template <typename Compute>
nb::object binary_or_not_implemented(nb::handle other, Compute compute) {
    std::optional<Operand> operand = maybe_operand(other);
    if (!operand) {
        return nb::borrow(Py_NotImplemented);
    }
    return nb::cast(compute(*operand));
}

Tensor to_anything(const Tensor& t, const nb::args& args, const nb::kwargs& kwargs) {
    Tensor result = t;
    const auto apply = [&](nb::handle target) {
        if (nb::isinstance<DType>(target)) {
            result = result.to(nb::cast<DType>(target));
        } else if (nb::isinstance<nb::str>(target)) {
            result = result.to(parse_device(nb::cast<std::string>(target)));
        } else if (nb::isinstance<Tensor>(target)) {
            const Tensor& like = nb::cast<const Tensor&>(target);
            result = result.to(like.device()).to(like.dtype());
        } else {
            throw TypeError("to() expects a dtype, a device name or a tensor");
        }
    };
    for (nb::handle arg : args) {
        apply(arg);
    }
    for (const char* key : {"device", "dtype"}) {
        if (kwargs.contains(key) && !nb::object(kwargs[key]).is_none()) {
            apply(kwargs[key]);
        }
    }
    return result;
}

Shape expand_target(const Tensor& t, const nb::args& args) {
    (void)t;
    return integers_from_args(args);
}

template <Tensor& (Tensor::*Method)(const Operand&)>
nb::object inplace(nb::handle self, nb::handle other) {
    Tensor& t = nb::cast<Tensor&>(self);
    std::optional<Operand> operand = maybe_operand(other);
    if (!operand) {
        return nb::borrow(Py_NotImplemented);
    }
    (t.*Method)(*operand);
    return nb::borrow(self);
}

} // namespace

nb::object tensor_item(const Tensor& t) {
    switch (t.dtype()) {
    case DType::F32:
    case DType::F64:
        return nb::float_(t.item());
    case DType::I64:
        return nb::int_(t.item<std::int64_t>());
    case DType::Bool:
        return nb::bool_(t.item<std::uint8_t>() != 0);
    }
    detail::unreachable();
}

nb::object tensor_to_list(const Tensor& t) {
    if (t.dim() == 0) {
        return tensor_item(t);
    }
    nb::list items;
    for (std::int64_t i = 0; i < t.size(0); ++i) {
        items.append(tensor_to_list(t.select(0, i)));
    }
    return items;
}

void bind_tensor(nb::module_& m) {
    nb::enum_<DType>(m, "DType", "Element type of a tensor.")
        .value("F32", DType::F32)
        .value("F64", DType::F64)
        .value("I64", DType::I64)
        .value("Bool", DType::Bool);

    nb::class_<Tensor>(m, "Tensor", "A multi-dimensional array with automatic differentiation.")
        .def(nb::init<const Tensor&>())
        .def_prop_ro("shape", [](const Tensor& t) { return shape_tuple(t.shape()); })
        .def_prop_ro("dtype", &Tensor::dtype)
        .def_prop_ro("device", [](const Tensor& t) { return device_name(t.device()); })
        .def_prop_ro("ndim", &Tensor::dim)
        .def_prop_ro("is_leaf", &Tensor::is_leaf)
        .def_prop_ro("grad", &Tensor::grad, "Accumulated gradient, or None.")
        .def_prop_ro("grad_fn",
                     [](const Tensor& t) -> std::optional<std::string> {
                         const auto function = t.grad_fn();
                         return function ? std::optional(function->name()) : std::nullopt;
                     })
        .def_prop_rw("requires_grad", &Tensor::requires_grad,
                     [](Tensor& t, bool required) { t.requires_grad_(required); })
        .def_prop_ro("T", &Tensor::T)
        .def_prop_ro("mT", &Tensor::mT)
        .def(
            "size",
            [](const Tensor& t, const nb::object& dim) -> nb::object {
                if (dim.is_none()) {
                    return shape_tuple(t.shape());
                }
                return nb::int_(t.size(nb::cast<std::int64_t>(dim)));
            },
            nb::arg("dim") = nb::none())
        .def("stride", [](const Tensor& t) { return shape_tuple(t.strides()); })
        .def("numel", &Tensor::numel)
        .def("item", &tensor_item, "The value of a tensor with one element.")
        .def("tolist", &tensor_to_list, "The values as nested Python lists.")
        .def("clone", &Tensor::clone, nb::call_guard<nb::gil_scoped_release>())
        .def("detach", &Tensor::detach)
        .def("detach_",
             [](nb::handle self) {
                 nb::cast<Tensor&>(self).detach_();
                 return nb::borrow(self);
             })
        .def("contiguous", &Tensor::contiguous)
        .def("is_contiguous", &Tensor::is_contiguous)
        .def("cuda", &Tensor::cuda)
        .def("cpu", &Tensor::cpu)
        .def("to", &to_anything, "Convert to a dtype and/or move to a device.")
        .def(
            "backward",
            [](const Tensor& t, const std::optional<Tensor>& gradient, bool retain_graph) {
                const nb::gil_scoped_release release;
                if (gradient) {
                    t.backward(*gradient, retain_graph);
                } else {
                    t.backward(retain_graph);
                }
            },
            nb::arg("gradient") = nb::none(), nb::arg("retain_graph") = false,
            "Compute gradients of this tensor with respect to the leaves of its graph.")
        .def("zero_grad", &Tensor::zero_grad)
        .def("retain_grad", &Tensor::retain_grad)
        .def(
            "requires_grad_",
            [](nb::handle self, bool required) {
                nb::cast<Tensor&>(self).requires_grad_(required);
                return nb::borrow(self);
            },
            nb::arg("requires_grad") = true)
        .def("_move_to_",
             [](Tensor& t, const std::string& device) { t.move_to_(parse_device(device)); })
        .def("_set_grad_device",
             [](Tensor& t, const std::string& device) {
                 if (const std::optional<Tensor> gradient = t.grad()) {
                     t.zero_grad();
                     const autograd::NoGradGuard guard;
                     t.impl()->autograd->grad = gradient->to(parse_device(device));
                 }
             })
        .def("__repr__",
             [](const Tensor& t) {
                 std::ostringstream text;
                 text << t;
                 return text.str();
             })
        .def("__len__",
             [](const Tensor& t) {
                 KAEL_CHECK_TYPE(t.dim() > 0, "len() of a 0-d tensor");
                 return t.size(0);
             })
        .def("__iter__",
             [](const Tensor& t) {
                 KAEL_CHECK_TYPE(t.dim() > 0, "iteration over a 0-d tensor");
                 nb::list items;
                 for (std::int64_t i = 0; i < t.size(0); ++i) {
                     items.append(t.select(0, i));
                 }
                 return nb::iter(items);
             })
        .def("__hash__",
             [](nb::handle self) { return reinterpret_cast<std::intptr_t>(self.ptr()); })
        .def("__bool__",
             [](const Tensor& t) {
                 KAEL_CHECK(t.numel() == 1,
                            "Boolean value of Tensor with more than one value is ambiguous");
                 return t.item() != 0.0;
             })
        .def("__float__", [](const Tensor& t) { return t.item(); })
        .def("__int__", [](const Tensor& t) { return static_cast<std::int64_t>(t.item()); })
        .def("__index__",
             [](const Tensor& t) {
                 KAEL_CHECK_TYPE(t.dtype() == DType::I64, "only integer tensors can be an index");
                 return t.item<std::int64_t>();
             })
        .def("__getitem__", &index_tensor, nb::arg("key").none())
        .def("__setitem__", &assign_index, nb::arg("key").none(), nb::arg("value"))
        .def("__neg__", [](const Tensor& t) { return kael::neg(t); })
        .def("__abs__", [](const Tensor& t) { return kael::abs(t); })
        .def("__invert__", [](const Tensor& t) { return kael::logical_not(t); })
        .def("__add__",
             [](const Tensor& a, nb::handle b) {
                 return binary_or_not_implemented(
                     b, [&](const Operand& o) { return kael::add(a, o); });
             })
        .def("__radd__",
             [](const Tensor& a, nb::handle b) {
                 return binary_or_not_implemented(
                     b, [&](const Operand& o) { return kael::add(o, a); });
             })
        .def("__sub__",
             [](const Tensor& a, nb::handle b) {
                 return binary_or_not_implemented(
                     b, [&](const Operand& o) { return kael::sub(a, o); });
             })
        .def("__rsub__",
             [](const Tensor& a, nb::handle b) {
                 return binary_or_not_implemented(
                     b, [&](const Operand& o) { return kael::sub(o, a); });
             })
        .def("__mul__",
             [](const Tensor& a, nb::handle b) {
                 return binary_or_not_implemented(
                     b, [&](const Operand& o) { return kael::mul(a, o); });
             })
        .def("__rmul__",
             [](const Tensor& a, nb::handle b) {
                 return binary_or_not_implemented(
                     b, [&](const Operand& o) { return kael::mul(o, a); });
             })
        .def("__truediv__",
             [](const Tensor& a, nb::handle b) {
                 return binary_or_not_implemented(
                     b, [&](const Operand& o) { return kael::div(a, o); });
             })
        .def("__rtruediv__",
             [](const Tensor& a, nb::handle b) {
                 return binary_or_not_implemented(
                     b, [&](const Operand& o) { return kael::div(o, a); });
             })
        .def("__floordiv__",
             [](const Tensor& a, nb::handle b) {
                 return binary_or_not_implemented(
                     b, [&](const Operand& o) { return kael::floor_divide(a, o); });
             })
        .def("__pow__",
             [](const Tensor& a, nb::handle b) {
                 return binary_or_not_implemented(
                     b, [&](const Operand& o) { return kael::pow(a, o); });
             })
        .def("__rpow__",
             [](const Tensor& a, nb::handle b) {
                 return binary_or_not_implemented(
                     b, [&](const Operand& o) { return kael::pow(o, a); });
             })
        .def(
            "__matmul__", [](const Tensor& a, const Tensor& b) { return kael::matmul(a, b); },
            nb::call_guard<nb::gil_scoped_release>())
        .def("__eq__",
             [](const Tensor& a, nb::handle b) {
                 return binary_or_not_implemented(b,
                                                  [&](const Operand& o) { return kael::eq(a, o); });
             })
        .def("__ne__",
             [](const Tensor& a, nb::handle b) {
                 return binary_or_not_implemented(b,
                                                  [&](const Operand& o) { return kael::ne(a, o); });
             })
        .def("__lt__",
             [](const Tensor& a, nb::handle b) {
                 return binary_or_not_implemented(b,
                                                  [&](const Operand& o) { return kael::lt(a, o); });
             })
        .def("__le__",
             [](const Tensor& a, nb::handle b) {
                 return binary_or_not_implemented(b,
                                                  [&](const Operand& o) { return kael::le(a, o); });
             })
        .def("__gt__",
             [](const Tensor& a, nb::handle b) {
                 return binary_or_not_implemented(b,
                                                  [&](const Operand& o) { return kael::gt(a, o); });
             })
        .def("__ge__",
             [](const Tensor& a, nb::handle b) {
                 return binary_or_not_implemented(b,
                                                  [&](const Operand& o) { return kael::ge(a, o); });
             })
        .def("__and__",
             [](const Tensor& a, nb::handle b) {
                 return binary_or_not_implemented(
                     b, [&](const Operand& o) { return kael::logical_and(a, o); });
             })
        .def("__or__",
             [](const Tensor& a, nb::handle b) {
                 return binary_or_not_implemented(
                     b, [&](const Operand& o) { return kael::logical_or(a, o); });
             })
        .def("__xor__",
             [](const Tensor& a, nb::handle b) {
                 return binary_or_not_implemented(
                     b, [&](const Operand& o) { return kael::logical_xor(a, o); });
             })
        .def("__iadd__", &inplace<&Tensor::add_>)
        .def("__isub__", &inplace<&Tensor::sub_>)
        .def("__imul__", &inplace<&Tensor::mul_>)
        .def("__itruediv__", &inplace<&Tensor::div_>)
        .def("__ipow__", &inplace<&Tensor::pow_>)
        .def("add_",
             [](nb::handle self, nb::handle other) {
                 nb::cast<Tensor&>(self).add_(operand_from(other));
                 return nb::borrow(self);
             })
        .def("sub_",
             [](nb::handle self, nb::handle other) {
                 nb::cast<Tensor&>(self).sub_(operand_from(other));
                 return nb::borrow(self);
             })
        .def("mul_",
             [](nb::handle self, nb::handle other) {
                 nb::cast<Tensor&>(self).mul_(operand_from(other));
                 return nb::borrow(self);
             })
        .def("div_",
             [](nb::handle self, nb::handle other) {
                 nb::cast<Tensor&>(self).div_(operand_from(other));
                 return nb::borrow(self);
             })
        .def("pow_",
             [](nb::handle self, nb::handle other) {
                 nb::cast<Tensor&>(self).pow_(operand_from(other));
                 return nb::borrow(self);
             })
        .def("copy_",
             [](nb::handle self, const Tensor& source) {
                 nb::cast<Tensor&>(self).copy_(source);
                 return nb::borrow(self);
             })
        .def("fill_",
             [](nb::handle self, nb::handle value) {
                 nb::cast<Tensor&>(self).fill_(scalar_from(value));
                 return nb::borrow(self);
             })
        .def("zero_",
             [](nb::handle self) {
                 nb::cast<Tensor&>(self).zero_();
                 return nb::borrow(self);
             })
        .def("clamp_",
             [](nb::handle self, nb::handle low, nb::handle high) {
                 nb::cast<Tensor&>(self).clamp_(scalar_from(low), scalar_from(high));
                 return nb::borrow(self);
             })
        .def("masked_fill_",
             [](nb::handle self, const Tensor& mask, nb::handle value) {
                 nb::cast<Tensor&>(self).masked_fill_(mask, scalar_from(value));
                 return nb::borrow(self);
             })
        .def("view",
             [](const Tensor& t, const nb::args& args) { return t.view(integers_from_args(args)); })
        .def("reshape", [](const Tensor& t,
                           const nb::args& args) { return t.reshape(integers_from_args(args)); })
        .def("expand",
             [](const Tensor& t, const nb::args& args) { return t.expand(expand_target(t, args)); })
        .def("permute", [](const Tensor& t,
                           const nb::args& args) { return t.permute(integers_from_args(args)); })
        .def("transpose", &Tensor::transpose)
        .def(
            "squeeze",
            [](const Tensor& t, const std::optional<std::int64_t>& dim) {
                return dim ? t.squeeze(*dim) : t.squeeze();
            },
            nb::arg("dim") = nb::none())
        .def("unsqueeze", &Tensor::unsqueeze)
        .def("narrow", &Tensor::narrow)
        .def("select", &Tensor::select)
        .def("flatten", &Tensor::flatten, nb::arg("start_dim") = 0, nb::arg("end_dim") = -1)
        .def("unflatten", &Tensor::unflatten)
        .def("split", &Tensor::split, nb::arg("split_size"), nb::arg("dim") = 0)
        .def("chunk", &Tensor::chunk, nb::arg("chunks"), nb::arg("dim") = 0);
}

} // namespace kael::python
