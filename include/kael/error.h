#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

namespace kael {

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

namespace detail {

[[noreturn]] inline void fail(const char* file, int line, const char* condition,
                              const std::string& message) {
    std::ostringstream os;
    os << "kael error at " << file << ":" << line << ": check '" << condition
       << "' failed: " << message;
    throw Error(os.str());
}

} // namespace detail

} // namespace kael

// `msg` is not parenthesized so that stream chains like "index " << i work.
#define KAEL_CHECK(cond, msg)                                                                      \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            ::std::ostringstream kael_check_stream_;                                               \
            kael_check_stream_ << msg;                                                             \
            ::kael::detail::fail(__FILE__, __LINE__, #cond, kael_check_stream_.str());             \
        }                                                                                          \
    } while (0)
