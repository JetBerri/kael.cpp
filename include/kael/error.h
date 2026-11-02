#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

namespace kael {

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class IndexError : public Error {
public:
    using Error::Error;
};

class ValueError : public Error {
public:
    using Error::Error;
};

class TypeError : public Error {
public:
    using Error::Error;
};

namespace detail {

template <typename Exception>
[[noreturn]] void fail(const char* file, int line, const char* condition,
                       const std::string& message) {
    std::ostringstream os;
    os << "kael error at " << file << ":" << line << ": check '" << condition
       << "' failed: " << message;
    throw Exception(os.str());
}

} // namespace detail

} // namespace kael

// `msg` is not parenthesized so that stream chains like "index " << i work.
// NOLINTBEGIN(bugprone-macro-parentheses)
#define KAEL_CHECK_AS(exception, cond, msg)                                                        \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            ::std::ostringstream kael_check_stream_;                                               \
            kael_check_stream_ << msg;                                                             \
            ::kael::detail::fail<exception>(__FILE__, __LINE__, #cond, kael_check_stream_.str());  \
        }                                                                                          \
    } while (0)
// NOLINTEND(bugprone-macro-parentheses)

#define KAEL_CHECK(cond, msg) KAEL_CHECK_AS(::kael::Error, cond, msg)
#define KAEL_CHECK_INDEX(cond, msg) KAEL_CHECK_AS(::kael::IndexError, cond, msg)
#define KAEL_CHECK_VALUE(cond, msg) KAEL_CHECK_AS(::kael::ValueError, cond, msg)
#define KAEL_CHECK_TYPE(cond, msg) KAEL_CHECK_AS(::kael::TypeError, cond, msg)
