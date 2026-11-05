#pragma once

#include <nanobind/nanobind.h>
#include <optional>
#include <string>
#include <vector>

#include "kael/kael.h"

namespace kael::python {

namespace nb = nanobind;

[[nodiscard]] std::string device_name(Device device);
[[nodiscard]] std::vector<std::int64_t> integers_from_args(const nb::args& args);
[[nodiscard]] Dims dims_from(const nb::object& dim);
[[nodiscard]] std::optional<Operand> maybe_operand(nb::handle value);
[[nodiscard]] Operand operand_from(nb::handle value);
[[nodiscard]] Scalar scalar_from(nb::handle value);

[[nodiscard]] Tensor tensor_from_python(nb::handle data, std::optional<DType> dtype);
[[nodiscard]] nb::object tensor_to_list(const Tensor& t);
[[nodiscard]] nb::object tensor_item(const Tensor& t);
[[nodiscard]] Tensor index_tensor(const Tensor& t, nb::handle key);
void assign_index(Tensor& t, nb::handle key, nb::handle value);

void bind_tensor(nb::module_& m);
void bind_ops(nb::module_& m);
void bind_autograd(nb::module_& m);
void bind_interop(nb::module_& m);
void bind_models(nb::module_& m);

} // namespace kael::python
