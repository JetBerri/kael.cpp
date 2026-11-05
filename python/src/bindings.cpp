#include <nanobind/stl/string.h>

#include "bind.h"

namespace kael::python {

std::string device_name(Device device) {
    return to_string(device);
}

std::vector<std::int64_t> integers_from_args(const nb::args& args) {
    std::vector<std::int64_t> values;
    if (args.size() == 1 && !nb::isinstance<nb::int_>(args[0])) {
        for (nb::handle item : nb::iter(args[0])) {
            values.push_back(nb::cast<std::int64_t>(item));
        }
        return values;
    }
    for (nb::handle item : args) {
        values.push_back(nb::cast<std::int64_t>(item));
    }
    return values;
}

Dims dims_from(const nb::object& dim) {
    if (dim.is_none()) {
        return {};
    }
    if (nb::isinstance<nb::int_>(dim)) {
        return {nb::cast<std::int64_t>(dim)};
    }
    Dims dims;
    for (nb::handle item : nb::iter(dim)) {
        dims.push_back(nb::cast<std::int64_t>(item));
    }
    return dims;
}

std::optional<Operand> maybe_operand(nb::handle value) {
    if (nb::isinstance<Tensor>(value)) {
        return Operand(nb::cast<const Tensor&>(value));
    }
    if (PyBool_Check(value.ptr())) {
        return Operand(value.ptr() == Py_True);
    }
    if (PyLong_Check(value.ptr())) {
        return Operand(nb::cast<std::int64_t>(value));
    }
    if (PyFloat_Check(value.ptr())) {
        return Operand(nb::cast<double>(value));
    }
    if (PyIndex_Check(value.ptr())) {
        return Operand(static_cast<std::int64_t>(PyNumber_AsSsize_t(value.ptr(), nullptr)));
    }
    if (PyObject_HasAttrString(value.ptr(), "__float__") != 0 &&
        PyObject_HasAttrString(value.ptr(), "__len__") == 0) {
        return Operand(nb::cast<double>(nb::float_(value)));
    }
    return std::nullopt;
}

Operand operand_from(nb::handle value) {
    std::optional<Operand> operand = maybe_operand(value);
    if (!operand) {
        throw TypeError("expected a tensor or a number, got " +
                        nb::cast<std::string>(nb::str(value.type())));
    }
    return *operand;
}

Scalar scalar_from(nb::handle value) {
    const Operand operand = operand_from(value);
    if (operand.is_tensor()) {
        throw TypeError("expected a number, got a tensor");
    }
    return operand.scalar();
}

} // namespace kael::python

NB_MODULE(_C, m) {
    namespace kp = kael::python;
    namespace nb = nanobind;

    nb::register_exception_translator([](const std::exception_ptr& error, void*) {
        try {
            std::rethrow_exception(error);
        } catch (const kael::IndexError& e) {
            PyErr_SetString(PyExc_IndexError, e.what());
        } catch (const kael::ValueError& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (const kael::TypeError& e) {
            PyErr_SetString(PyExc_TypeError, e.what());
        } catch (const kael::Error& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

    m.attr("__version__") = std::string(kael::version());

    kp::bind_tensor(m);
    kp::bind_ops(m);
    kp::bind_autograd(m);
    kp::bind_interop(m);
    kp::bind_models(m);
}
