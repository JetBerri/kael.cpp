#pragma once

#include <stdexcept>
#include <sstream>
#include <string>

namespace kael{

    class Error : public std::runtime_error{ // inherited from runtime_error
        public:
            using std::runtime_error::runtime_error;

    };

namespace detail {
    [[noreturn]] inline void fail(const char* file, int line,
                                  const char* expr, const std::string& msg) {

                                std::ostringstream os;
                                os << "kael error at " << file << ":" << line
                                    << ": check '" << expr << "' failed: " << msg; // msg debug
                                    
                                throw Error(os.str()); // throw error
                                }
    }

}

// macro rule
#define KAEL_CHECK(cond, msg) \ 
    do { \
       if (!(cond)){ \
            ::kael::detail::fail(__FILE__, __LINE__, #cond, (msg)); \
        } \
    } while(0)
// use: KAEL_CHECK(!name.empty(), "empty")