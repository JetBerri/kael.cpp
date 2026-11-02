cmake_minimum_required(VERSION 3.24)
project(kael VERSION 0.1.0 LANGUAGES CXX)

option(KAEL_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(KAEL_ENABLE_SANITIZERS "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

# Applies the project warning flags (and sanitizers, if enabled) to a target.
# Always PRIVATE: consumers of kael do not inherit our flags.
function(kael_configure_target target)
    set(gcc_like "$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>")
    target_compile_options(${target} PRIVATE
        $<${gcc_like}:-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion>
    )
    if (KAEL_WARNINGS_AS_ERRORS)
        target_compile_options(${target} PRIVATE $<${gcc_like}:-Werror>)
    endif ()
    if (KAEL_ENABLE_SANITIZERS)
        set(san_flags -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_compile_options(${target} PRIVATE $<${gcc_like}:${san_flags}>)
        target_link_options(${target} PRIVATE $<${gcc_like}:${san_flags}>)
    endif ()
endfunction()

add_library(kael
    src/shape.cpp
    src/storage.cpp
    src/tensor_impl.cpp
    src/tensor.cpp
    src/factory.cpp
    src/random.cpp
)

add_library(kael::kael ALIAS kael)

target_include_directories(kael PUBLIC include PRIVATE src)
target_compile_features(kael PUBLIC cxx_std_20)
kael_configure_target(kael)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_subdirectory(tests)
