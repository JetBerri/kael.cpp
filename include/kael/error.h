#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

namespace kael {

// Base exception for all errors raised by kael.
class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

namespace detail {

// Builds the full error message and throws kael::Error. Never returns.
[[noreturn]] inline void fail(const char* file, int line,
                              const char* expr, const std::string& msg) {
    std::ostringstream os;
    os << "kael error at " << file << ":" << line
       << ": check '" << expr << "' failed: " << msg;
    throw Error(os.str());
}

} // namespace detail

} // namespace kael

// Throws kael::Error if `cond` is false.
// `msg` is intentionally not parenthesized so that stream chains
// like "index " << i << " out of range" work.
//
// Usage:
//   KAEL_CHECK(!name.empty(), "empty name");
//   KAEL_CHECK(i < n, "index " << i << " out of range [0, " << n << ")");
#define KAEL_CHECK(cond, msg)                                    \
    do {                                                         \
        if (!(cond)) {                                           \
            ::std::ostringstream kael_check_stream_;             \
            kael_check_stream_ << msg;                           \
            ::kael::detail::fail(__FILE__, __LINE__, #cond,      \
                                 kael_check_stream_.str());      \
        }                                                        \
    } while (0)