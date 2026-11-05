#include <memory>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <utility>

#include "kael/data/mnist.h"

#include "bind.h"

namespace kael::python {

namespace {

struct ParsedKey {
    std::vector<nb::object> items;
    bool has_ellipsis = false;
};

ParsedKey parse_key(nb::handle key) {
    ParsedKey parsed;
    if (nb::isinstance<nb::tuple>(key)) {
        for (nb::handle item : nb::iter(key)) {
            parsed.items.push_back(nb::borrow(item));
        }
    } else {
        parsed.items.push_back(nb::borrow(key));
    }
    for (const nb::object& item : parsed.items) {
        if (item.ptr() == Py_Ellipsis) {
            KAEL_CHECK_INDEX(!parsed.has_ellipsis, "an index can only have one ellipsis");
            parsed.has_ellipsis = true;
        }
    }
    return parsed;
}

std::int64_t consumed_dims(const ParsedKey& key) {
    std::int64_t consumed = 0;
    for (const nb::object& item : key.items) {
        if (!item.is_none() && item.ptr() != Py_Ellipsis) {
            ++consumed;
        }
    }
    return consumed;
}

bool is_integer_key(nb::handle item) {
    return PyIndex_Check(item.ptr()) != 0 && !nb::isinstance<Tensor>(item);
}

Tensor boolean_mask(const Tensor& t, const Tensor& mask) {
    KAEL_CHECK_INDEX(mask.shape() == t.shape(), "boolean mask of shape "
                                                    << to_string(mask.shape())
                                                    << " does not match tensor of shape "
                                                    << to_string(t.shape()));
    const Tensor host = mask.cpu().contiguous();
    const auto* flags = host.data_ptr<std::uint8_t>();
    std::vector<std::int64_t> positions;
    for (std::int64_t i = 0; i < host.numel(); ++i) {
        if (flags[i] != 0) {
            positions.push_back(i);
        }
    }
    const Tensor indices = from_vector(positions).to(t.device());
    return kael::index_select(t.reshape({-1}), 0, indices);
}

Tensor apply_key(const Tensor& t, nb::handle key) {
    const ParsedKey parsed = parse_key(key);
    const std::int64_t consumed = consumed_dims(parsed);
    KAEL_CHECK_INDEX(consumed <= t.dim(), "too many indices for a tensor of dimension " << t.dim());

    Tensor view = t;
    std::int64_t dim = 0;
    for (const nb::object& item : parsed.items) {
        if (item.ptr() == Py_Ellipsis) {
            dim += t.dim() - consumed;
        } else if (item.is_none()) {
            view = view.unsqueeze(dim);
            ++dim;
        } else if (PySlice_Check(item.ptr())) {
            Py_ssize_t start = 0;
            Py_ssize_t stop = 0;
            Py_ssize_t step = 0;
            if (PySlice_Unpack(item.ptr(), &start, &stop, &step) < 0) {
                throw nb::python_error();
            }
            KAEL_CHECK_VALUE(step > 0, "negative step not supported in a slice, got " << step);
            const Py_ssize_t size = view.size(dim);
            const Py_ssize_t length = PySlice_AdjustIndices(size, &start, &stop, step);
            view = length > 0 ? view.slice(dim, start, stop, step) : view.narrow(dim, 0, 0);
            ++dim;
        } else if (nb::isinstance<Tensor>(item)) {
            const Tensor& indices = nb::cast<const Tensor&>(item);
            if (indices.dtype() == DType::Bool && parsed.items.size() == 1) {
                return boolean_mask(view, indices);
            }
            KAEL_CHECK_TYPE(indices.dtype() == DType::I64,
                            "tensor indices must be int64 or a whole-tensor bool mask, got "
                                << indices.dtype());
            if (indices.dim() == 0) {
                view = view.select(dim, indices.item<std::int64_t>());
            } else {
                KAEL_CHECK_INDEX(indices.dim() == 1, "tensor indices must be 1-d, got shape "
                                                         << to_string(indices.shape()));
                view = kael::index_select(view, dim, indices);
                ++dim;
            }
        } else if (PyBool_Check(item.ptr()) != 0) {
            throw IndexError("a bool is not a valid index; use a bool tensor as a mask");
        } else if (is_integer_key(item)) {
            view = view.select(dim, nb::cast<std::int64_t>(nb::int_(item)));
        } else {
            throw IndexError("unsupported index of type " +
                             nb::cast<std::string>(nb::str(item.type())));
        }
    }
    return view;
}

DType dtype_from_numpy_kind(const std::string& kind, std::int64_t size) {
    if (kind == "f" && size == 4) {
        return DType::F32;
    }
    if (kind == "f" && size == 8) {
        return DType::F64;
    }
    if (kind == "i" && size == 8) {
        return DType::I64;
    }
    if (kind == "b") {
        return DType::Bool;
    }
    throw TypeError("numpy dtype of kind '" + kind + "' and size " + std::to_string(size) +
                    " is not supported: kael has float32, float64, int64 and bool");
}

void numpy_owner_release(nb::object* array) {
    if (Py_IsInitialized() != 0) {
        const nb::gil_scoped_acquire acquire;
        delete array;
    }
}

Tensor from_numpy(const nb::object& array) {
    const nb::object numpy_dtype = array.attr("dtype");
    const DType dtype = dtype_from_numpy_kind(nb::cast<std::string>(numpy_dtype.attr("kind")),
                                              nb::cast<std::int64_t>(numpy_dtype.attr("itemsize")));
    KAEL_CHECK_TYPE(nb::cast<bool>(numpy_dtype.attr("isnative")),
                    "numpy arrays must use the native byte order");

    const nb::dict interface = nb::cast<nb::dict>(array.attr("__array_interface__"));
    const nb::tuple data = nb::cast<nb::tuple>(interface["data"]);
    auto* first = reinterpret_cast<std::byte*>(nb::cast<std::uintptr_t>(data[0]));
    const bool read_only = nb::cast<bool>(data[1]);

    Shape shape;
    Strides strides;
    const std::int64_t item_size = element_size(dtype);
    const nb::tuple numpy_shape = nb::cast<nb::tuple>(array.attr("shape"));
    const nb::tuple numpy_strides = nb::cast<nb::tuple>(array.attr("strides"));
    std::int64_t last_byte = item_size;
    bool empty_array = false;
    for (std::size_t i = 0; i < numpy_shape.size(); ++i) {
        const auto size = nb::cast<std::int64_t>(numpy_shape[i]);
        const auto stride = nb::cast<std::int64_t>(numpy_strides[i]);
        KAEL_CHECK_VALUE(stride >= 0, "numpy arrays with negative strides are not supported; "
                                      "call .copy() first");
        KAEL_CHECK_VALUE(stride % item_size == 0,
                         "numpy strides must be a multiple of the element size");
        shape.push_back(size);
        strides.push_back(stride / item_size);
        empty_array = empty_array || size == 0;
        if (size > 0) {
            last_byte += (size - 1) * stride;
        }
    }
    KAEL_CHECK(reinterpret_cast<std::uintptr_t>(first) % static_cast<std::uintptr_t>(item_size) ==
                   0,
               "numpy array is not aligned to its element size");

    std::shared_ptr<void> owner(new nb::object(array), [](void* held) {
        numpy_owner_release(static_cast<nb::object*>(held));
    });
    auto impl = std::make_shared<TensorImpl>();
    impl->storage = std::make_shared<Storage>(
        Storage::external(first, empty_array ? 0 : last_byte, std::move(owner)));
    impl->shape = std::move(shape);
    impl->strides = std::move(strides);
    impl->dtype = dtype;
    Tensor result = Tensor::from_impl(std::move(impl));
    return read_only ? result.clone() : result;
}

nb::dlpack::dtype dlpack_dtype(DType dtype) {
    switch (dtype) {
    case DType::F32:
        return nb::dtype<float>();
    case DType::F64:
        return nb::dtype<double>();
    case DType::I64:
        return nb::dtype<std::int64_t>();
    case DType::Bool:
        return nb::dtype<bool>();
    }
    detail::unreachable();
}

template <typename Framework> nb::ndarray<Framework> as_ndarray(const Tensor& t) {
    KAEL_CHECK(t.device() == Device::CPU, "cannot share a tensor on device "
                                              << t.device() << " with numpy; call .cpu() first");
    KAEL_CHECK(!t.requires_grad(),
               "cannot share a tensor that requires grad; call .detach() first");
    const nb::capsule owner(
        new std::shared_ptr<Storage>(t.impl()->storage),
        [](void* held) noexcept { delete static_cast<std::shared_ptr<Storage>*>(held); });
    std::vector<std::size_t> shape(t.shape().begin(), t.shape().end());
    std::vector<std::int64_t> strides(t.strides().begin(), t.strides().end());
    return nb::ndarray<Framework>(t.device_data(), shape.size(), shape.data(), owner,
                                  strides.data(), dlpack_dtype(t.dtype()));
}

} // namespace

Tensor index_tensor(const Tensor& t, nb::handle key) {
    return apply_key(t, key);
}

void assign_index(Tensor& t, nb::handle key, nb::handle value) {
    const ParsedKey parsed = parse_key(key);
    if (parsed.items.size() == 1 && nb::isinstance<Tensor>(parsed.items.front()) &&
        nb::cast<const Tensor&>(parsed.items.front()).dtype() == DType::Bool) {
        KAEL_CHECK_TYPE(!nb::isinstance<Tensor>(value),
                        "assigning through a bool mask needs a number");
        t.masked_fill_(nb::cast<const Tensor&>(parsed.items.front()), scalar_from(value));
        return;
    }
    for (const nb::object& item : parsed.items) {
        if (nb::isinstance<Tensor>(item) && nb::cast<const Tensor&>(item).dim() > 0) {
            throw TypeError("assigning through a tensor index is not supported: it selects a copy");
        }
    }
    Tensor view = apply_key(t, key);
    if (nb::isinstance<Tensor>(value)) {
        view.copy_(nb::cast<const Tensor&>(value));
    } else {
        view.fill_(scalar_from(value));
    }
}

namespace {

constexpr std::size_t kMaxNesting = 32;

struct Flattened {
    Shape shape;
    std::optional<std::size_t> leaf_depth;
    std::vector<double> reals;
    std::vector<std::int64_t> integers;
    bool has_float = false;
    bool has_integer = false;
    bool has_bool = false;
};

std::int64_t sequence_integer(const Tensor& t) {
    if (is_floating_point(t.dtype())) {
        return 0;
    }
    return t.dtype() == DType::I64 ? t.item<std::int64_t>() : static_cast<std::int64_t>(t.item());
}

void flatten_python(nb::handle data, Flattened& out, std::size_t depth) {
    if (nb::isinstance<nb::list>(data) || nb::isinstance<nb::tuple>(data)) {
        KAEL_CHECK_VALUE(depth < kMaxNesting,
                         "sequences nested deeper than " << kMaxNesting << " levels");
        KAEL_CHECK_VALUE(!out.leaf_depth || depth < *out.leaf_depth,
                         "nested sequences have inconsistent depth");
        const auto length = static_cast<std::int64_t>(nb::len(data));
        if (depth == out.shape.size()) {
            out.shape.push_back(length);
        } else {
            KAEL_CHECK_VALUE(out.shape[depth] == length,
                             "nested sequences have inconsistent sizes: "
                                 << out.shape[depth] << " and " << length << " at depth " << depth);
        }
        for (nb::handle item : nb::iter(data)) {
            flatten_python(item, out, depth + 1);
        }
        return;
    }
    KAEL_CHECK_VALUE(depth == out.shape.size() && (!out.leaf_depth || depth == *out.leaf_depth),
                     "nested sequences have inconsistent depth");
    out.leaf_depth = depth;
    if (nb::isinstance<Tensor>(data)) {
        const Tensor& t = nb::cast<const Tensor&>(data);
        KAEL_CHECK_VALUE(t.dim() == 0, "only 0-d tensors can be elements of a nested sequence");
        out.reals.push_back(t.item());
        out.integers.push_back(sequence_integer(t));
        out.has_float = out.has_float || is_floating_point(t.dtype());
        out.has_integer = out.has_integer || t.dtype() == DType::I64;
        out.has_bool = out.has_bool || t.dtype() == DType::Bool;
        return;
    }
    if (PyBool_Check(data.ptr())) {
        out.has_bool = true;
        const bool value = data.ptr() == Py_True;
        out.reals.push_back(value ? 1.0 : 0.0);
        out.integers.push_back(value ? 1 : 0);
    } else if (PyLong_Check(data.ptr()) || PyIndex_Check(data.ptr())) {
        out.has_integer = true;
        const auto value = nb::cast<std::int64_t>(nb::int_(data));
        out.reals.push_back(static_cast<double>(value));
        out.integers.push_back(value);
    } else if (PyFloat_Check(data.ptr()) || PyObject_HasAttrString(data.ptr(), "__float__") != 0) {
        out.has_float = true;
        const double value = nb::cast<double>(nb::float_(data));
        out.reals.push_back(value);
        out.integers.push_back(0);
    } else {
        throw TypeError("cannot build a tensor from an element of type " +
                        nb::cast<std::string>(nb::str(data.type())));
    }
}

} // namespace

Tensor tensor_from_python(nb::handle data, std::optional<DType> dtype) {
    if (nb::isinstance<Tensor>(data)) {
        const Tensor copy = nb::cast<const Tensor&>(data).detach().clone();
        return dtype ? copy.to(*dtype) : copy;
    }
    if (PyObject_HasAttrString(data.ptr(), "__array_interface__") != 0) {
        const Tensor copy = from_numpy(nb::borrow(data)).clone();
        return dtype ? copy.to(*dtype) : copy;
    }
    Flattened flat;
    flatten_python(data, flat, 0);
    const DType inferred = flat.has_float     ? DType::F32
                           : flat.has_integer ? DType::I64
                           : flat.has_bool    ? DType::Bool
                                              : DType::F32;
    const DType target = dtype.value_or(inferred);
    return flat.has_float ? from_vector(flat.reals, flat.shape).to(target)
                          : from_vector(flat.integers, flat.shape).to(target);
}

void bind_interop(nb::module_& m) {
    m.def(
        "tensor", &tensor_from_python, nb::arg("data"), nb::arg("dtype") = nb::none(),
        "Build a tensor from nested Python sequences, numbers or a numpy array, copying the data.");
    m.def("from_numpy", &from_numpy, nb::arg("array"),
          "View a numpy array as a tensor without copying; the array stays alive with the tensor.");

    auto tensor = nb::borrow<nb::class_<Tensor>>(m.attr("Tensor"));
    tensor.def(
        "numpy", [](const Tensor& t) { return as_ndarray<nb::numpy>(t); },
        "View this tensor as a numpy array without copying.");
    tensor.def(
        "__array__",
        [](const Tensor& t, const nb::object& dtype, const nb::object& copy) -> nb::object {
            nb::object array = nb::cast(as_ndarray<nb::numpy>(t));
            if (!dtype.is_none()) {
                array = array.attr("astype")(dtype);
            }
            if (!copy.is_none() && nb::cast<bool>(copy)) {
                array = array.attr("copy")();
            }
            return array;
        },
        nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none());
    tensor.def("__dlpack__", [](const Tensor& t, const nb::kwargs& kwargs) {
        return nb::cast(as_ndarray<nb::numpy>(t)).attr("__dlpack__")(**kwargs);
    });
    tensor.def("__dlpack_device__", [](const Tensor& t) {
        KAEL_CHECK(t.device() == Device::CPU, "DLPack exchange is only available for cpu tensors");
        return nb::make_tuple(1, 0);
    });

    m.def("_load_mnist", [](const std::string& directory, bool train) {
        const auto part = train ? data::MnistPart::Train : data::MnistPart::Test;
        const data::MnistSplit split = data::load_mnist(directory, part);
        return nb::make_tuple(split.images, split.labels);
    });
}

} // namespace kael::python
