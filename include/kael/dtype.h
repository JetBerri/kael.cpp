#pragma once

#include <cstdint>
#include <ostream>
#include <type_traits>

#include "kael/error.h"

namespace kael {

namespace detail {

[[noreturn]] inline void unreachable() {
    throw Error("kael internal error: reached unreachable code");
}

} // namespace detail

// Bool is stored as std::uint8_t: sizeof(bool) is not fixed by the standard and
// std::vector<bool> cannot hand out element pointers.
enum class DType { F32, F64, I64, Bool };

enum class Device { CPU, CUDA };

constexpr std::int64_t element_size(DType dtype) {
    switch (dtype) {
    case DType::F32:
        return 4;
    case DType::F64:
    case DType::I64:
        return 8;
    case DType::Bool:
        return 1;
    }
    detail::unreachable();
}

constexpr const char* to_string(DType dtype) {
    switch (dtype) {
    case DType::F32:
        return "float32";
    case DType::F64:
        return "float64";
    case DType::I64:
        return "int64";
    case DType::Bool:
        return "bool";
    }
    detail::unreachable();
}

constexpr bool is_floating_point(DType dtype) {
    return dtype == DType::F32 || dtype == DType::F64;
}

constexpr DType promote_types(DType a, DType b) {
    constexpr auto rank = [](DType dtype) {
        switch (dtype) {
        case DType::Bool:
            return 0;
        case DType::I64:
            return 1;
        case DType::F32:
            return 2;
        case DType::F64:
            return 3;
        }
        detail::unreachable();
    };
    return rank(a) >= rank(b) ? a : b;
}

inline std::ostream& operator<<(std::ostream& os, DType dtype) {
    return os << to_string(dtype);
}

constexpr const char* to_string(Device device) {
    switch (device) {
    case Device::CPU:
        return "cpu";
    case Device::CUDA:
        return "cuda";
    }
    detail::unreachable();
}

inline std::ostream& operator<<(std::ostream& os, Device device) {
    return os << to_string(device);
}

template <typename T>
struct dtype_of;

template <>
struct dtype_of<float> {
    static constexpr DType value = DType::F32;
};
template <>
struct dtype_of<double> {
    static constexpr DType value = DType::F64;
};
template <>
struct dtype_of<std::int64_t> {
    static constexpr DType value = DType::I64;
};
template <>
struct dtype_of<std::uint8_t> {
    static constexpr DType value = DType::Bool;
};

template <typename T>
inline constexpr DType dtype_of_v = dtype_of<T>::value;

template <DType D>
struct cpp_type;

template <>
struct cpp_type<DType::F32> {
    using type = float;
};
template <>
struct cpp_type<DType::F64> {
    using type = double;
};
template <>
struct cpp_type<DType::I64> {
    using type = std::int64_t;
};
template <>
struct cpp_type<DType::Bool> {
    using type = std::uint8_t;
};

template <DType D>
using cpp_type_t = typename cpp_type<D>::type;

// A plain static_cast would turn 2.5 into the invalid bool value 2.
template <typename To, typename From>
constexpr To convert_value(From value) {
    if constexpr (std::is_same_v<To, std::uint8_t>) {
        return value != From{0} ? To{1} : To{0};
    } else {
        return static_cast<To>(value);
    }
}

template <typename F>
decltype(auto) dispatch(DType dtype, F&& f) {
    switch (dtype) {
    case DType::F32:
        return f.template operator()<float>();
    case DType::F64:
        return f.template operator()<double>();
    case DType::I64:
        return f.template operator()<std::int64_t>();
    case DType::Bool:
        return f.template operator()<std::uint8_t>();
    }
    detail::unreachable();
}

template <typename F>
decltype(auto) dispatch_floating(DType dtype, const char* op, F&& f) {
    KAEL_CHECK(is_floating_point(dtype),
               op << " only supports floating point dtypes, got " << dtype);
    if (dtype == DType::F32) {
        return f.template operator()<float>();
    }
    return f.template operator()<double>();
}

template <typename F>
decltype(auto) dispatch_no_bool(DType dtype, const char* op, F&& f) {
    KAEL_CHECK(dtype != DType::Bool, op << " does not support dtype bool");
    if (dtype == DType::F32) {
        return f.template operator()<float>();
    }
    if (dtype == DType::F64) {
        return f.template operator()<double>();
    }
    return f.template operator()<std::int64_t>();
}

} // namespace kael
