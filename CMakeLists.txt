cmake_minimum_required(VERSION 3.24)
project(kael VERSION 0.1.0 LANGUAGES CXX)

option(KAEL_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(KAEL_ENABLE_SANITIZERS "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(KAEL_BUILD_BENCHMARKS "Build the benchmarks (use a Release build)" OFF)

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
    src/print.cpp
    src/copy.cpp
    src/view.cpp
    src/iterator.cpp
    src/ops/binary.cpp
    src/ops/unary.cpp
    src/ops/inplace.cpp
    src/ops/reduce.cpp
    src/ops/linalg.cpp
    src/ops/indexing.cpp
    src/ops/nn.cpp
    src/autograd/grad_mode.cpp
    src/autograd/tensor_autograd.cpp
    src/autograd/node.cpp
    src/autograd/engine.cpp
    src/autograd/recording.cpp
    src/autograd/functions_math.cpp
    src/autograd/functions_reduce.cpp
    src/autograd/functions_view.cpp
    src/autograd/functions_indexing.cpp
    src/autograd/functions_linalg.cpp
    src/autograd/gradcheck.cpp
    src/nn/module.cpp
    src/nn/init.cpp
    src/nn/functional.cpp
    src/nn/loss.cpp
    src/nn/layers.cpp
    src/optim/optimizer.cpp
    src/version.cpp
)

add_library(kael::kael ALIAS kael)

target_include_directories(kael PUBLIC include PRIVATE src)
target_compile_features(kael PUBLIC cxx_std_20)
target_compile_definitions(kael PRIVATE KAEL_VERSION="${PROJECT_VERSION}")
kael_configure_target(kael)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_subdirectory(tests)

if (KAEL_BUILD_BENCHMARKS)
    if (KAEL_ENABLE_SANITIZERS)
        message(FATAL_ERROR "Sanitizers distort timings: do not combine them with benchmarks")
    endif ()
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        benchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.9.0.tar.gz
    )
    FetchContent_MakeAvailable(benchmark)
    add_subdirectory(benchmarks)
endif ()
