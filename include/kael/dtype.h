#pragma once

#include <cstdint>
#include <ostream>
#include <type_traits>
#include <utility>

#include "kael/error.h"

namespace kael {

namespace detail {

// Marks code that a complete switch over an enum can never reach. The
// compiler cannot prove the switch is exhaustive, so without this it warns
// about a missing return. (std::unreachable is C++23; kael is C++20.)
[[noreturn]] inline void unreachable() {
    throw Error("kael internal error: reached unreachable code");
}

} // namespace detail

// Element type of a tensor, chosen at run time (see docs/adr/0001).
//
// Bool is stored as one std::uint8_t per element, holding 0 or 1:
//   - sizeof(bool) is not fixed by the standard;
//   - std::vector<bool> packs bits and cannot hand out element pointers.
// So the C++ type behind DType::Bool is std::uint8_t, not bool.
enum class DType { F32, F64, I64, Bool };

// Where the bytes of a tensor live. Only CPU is implemented for now.
enum class Device { CPU, CUDA };

// All switches below have no `default`: adding a DType makes -Wall point at
// every switch that does not handle it yet.

// Bytes per element.
constexpr std::int64_t element_size(DType dtype) {
    switch (dtype) {
    case DType::F32:
        return 4;
    case DType::F64:
        return 8;
    case DType::I64:
        return 8;
    case DType::Bool:
        return 1;
    }
    detail::unreachable();
}

// Name of the dtype as PyTorch spells it, for error messages.
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

// True for F32 and F64.
constexpr bool is_floating_point(DType dtype) {
    switch (dtype) {
    case DType::F32:
    case DType::F64:
        return true;
    case DType::I64:
    case DType::Bool:
        return false;
    }
    detail::unreachable();
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

// ---------------------------------------------------------------------------
// C++ type -> DType, at compile time.
//
//   static_assert(dtype_of_v<float> == DType::F32);
//   dtype_of_v<int>   // does not compile: unsupported type
// ---------------------------------------------------------------------------

template <typename T>
struct dtype_of; // no definition: unsupported types fail to compile

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

// ---------------------------------------------------------------------------
// DType -> C++ type, for a DType known at compile time.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Value conversion between element types.
//
// Same as static_cast, except when converting to Bool (std::uint8_t): any
// non-zero value becomes 1, so 2.5 -> true and NaN -> true (NaN != 0).
// Floating point -> integer truncates towards zero (2.7 -> 2, -2.7 -> -2).
// Converting NaN or infinity to an integer is undefined behaviour in C++;
// kael does not check it (it would cost a branch per element), so the
// result is unspecified.
// ---------------------------------------------------------------------------

template <typename To, typename From>
constexpr To convert_value(From value) {
    if constexpr (std::is_same_v<To, std::uint8_t>) {
        return value != From{0} ? To{1} : To{0};
    } else {
        return static_cast<To>(value);
    }
}

// ---------------------------------------------------------------------------
// Dispatch: run code with the C++ type that matches a run-time DType.
//
// `f` is a lambda with a template type parameter (C++20):
//
//   auto bytes = kael::dispatch(dtype, []<typename T>() { return sizeof(T); });
//
// Every branch must return the same type. Writing each operation as one such
// lambda is what lets kael implement it once for every dtype.
// ---------------------------------------------------------------------------

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

// Like dispatch, but only for F32 and F64. Throws for any other dtype, naming
// the operation `op` in the message.
template <typename F>
decltype(auto) dispatch_floating(DType dtype, const char* op, F&& f) {
    KAEL_CHECK(is_floating_point(dtype),
               op << " only supports floating point dtypes, got " << dtype);
    if (dtype == DType::F32) {
        return f.template operator()<float>();
    }
    return f.template operator()<double>();
}

// Like dispatch, but rejects Bool (arithmetic that makes no sense on bools).
template <typename F>
decltype(auto) dispatch_no_bool(DType dtype, const char* op, F&& f) {
    KAEL_CHECK(dtype != DType::Bool, op << " does not support dtype bool");
    switch (dtype) {
    case DType::F32:
        return f.template operator()<float>();
    case DType::F64:
        return f.template operator()<double>();
    case DType::I64:
        return f.template operator()<std::int64_t>();
    case DType::Bool:
        break; // rejected by the check above
    }
    detail::unreachable();
}

} // namespace kael
