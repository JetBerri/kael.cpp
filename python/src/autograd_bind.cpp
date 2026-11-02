#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "bind.h"

namespace kael::python {

void bind_autograd(nb::module_& m) {
    m.def("is_grad_enabled", &autograd::GradMode::is_enabled);
    m.def("set_grad_enabled", &autograd::GradMode::set_enabled);
    m.def(
        "gradcheck",
        [](const std::function<Tensor(const std::vector<Tensor>&)>& function,
           const std::vector<Tensor>& inputs) {
            const autograd::GradcheckResult result = autograd::gradcheck(function, inputs);
            return nb::make_tuple(result.passed, result.report);
        },
        nb::arg("function"), nb::arg("inputs"),
        "Compare the gradients of function(inputs) with a numerical estimate.");
}

} // namespace kael::python
