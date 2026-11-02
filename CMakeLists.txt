cmake_minimum_required(VERSION 3.24)
project(kael VERSION 0.4.0 LANGUAGES CXX)

option(KAEL_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(KAEL_ENABLE_SANITIZERS "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(KAEL_ENABLE_TSAN "Build with ThreadSanitizer" OFF)
option(KAEL_BUILD_BENCHMARKS "Build the benchmarks (use a Release build)" OFF)
option(KAEL_BUILD_EXAMPLES "Build the examples" OFF)
option(KAEL_USE_BLAS "Multiply matrices with the system BLAS" OFF)
option(KAEL_ENABLE_CUDA "Build the CUDA backend" OFF)
option(KAEL_FAKE_CUDA "Run the CUDA kernels on the host, to test them without a GPU" OFF)
option(KAEL_USE_CUBLAS "Multiply matrices on the device with cuBLAS" OFF)
option(KAEL_BUILD_TESTS "Build the C++ tests" ON)
option(KAEL_BUILD_PYTHON "Build the Python extension module" OFF)
option(KAEL_NATIVE "Optimize for the CPU of the build machine" OFF)

if (KAEL_ENABLE_SANITIZERS AND KAEL_ENABLE_TSAN)
    message(FATAL_ERROR "ThreadSanitizer cannot be combined with AddressSanitizer")
endif ()

if (KAEL_ENABLE_CUDA AND KAEL_FAKE_CUDA)
    message(FATAL_ERROR "KAEL_ENABLE_CUDA and KAEL_FAKE_CUDA are mutually exclusive")
endif ()

if (KAEL_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(kael_device_backend src/cuda/backend.cu)
elseif (KAEL_FAKE_CUDA)
    set(kael_device_backend src/cuda/backend_fake.cpp)
else ()
    set(kael_device_backend src/cuda/backend_none.cpp)
endif ()

function(kael_configure_target target)
    set(gcc_family "$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>")
    set(gcc_like "$<AND:$<COMPILE_LANGUAGE:CXX>,${gcc_family}>")
    target_compile_options(${target} PRIVATE
        $<${gcc_like}:-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion>
    )
    if (KAEL_WARNINGS_AS_ERRORS)
        target_compile_options(${target} PRIVATE $<${gcc_like}:-Werror>)
    endif ()
    if (KAEL_ENABLE_SANITIZERS)
        set(san_flags -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_compile_options(${target} PRIVATE $<${gcc_like}:${san_flags}>)
        target_link_options(${target} PRIVATE $<${gcc_family}:${san_flags}>)
    endif ()
    if (KAEL_ENABLE_TSAN)
        set(tsan_flags -fsanitize=thread -fno-omit-frame-pointer)
        target_compile_options(${target} PRIVATE $<${gcc_like}:${tsan_flags}>)
        target_link_options(${target} PRIVATE $<${gcc_family}:${tsan_flags}>)
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
    src/parallel.cpp
    src/allocator.cpp
    src/device.cpp
    ${kael_device_backend}
    src/linalg/matmul_versions.cpp
    src/ops/binary.cpp
    src/ops/unary.cpp
    src/ops/inplace.cpp
    src/ops/reduce.cpp
    src/ops/linalg.cpp
    src/ops/indexing.cpp
    src/ops/nn.cpp
    src/ops/sequence.cpp
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
    src/data/dataset.cpp
    src/data/mnist.cpp
    src/version.cpp
)

add_library(kael::kael ALIAS kael)

target_include_directories(kael PUBLIC include PRIVATE src)
target_compile_features(kael PUBLIC cxx_std_20)
target_compile_definitions(kael PRIVATE KAEL_VERSION="${PROJECT_VERSION}")
kael_configure_target(kael)

if (KAEL_BUILD_PYTHON)
    set_target_properties(kael PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif ()

if (KAEL_ENABLE_SANITIZERS OR KAEL_ENABLE_TSAN)
    target_compile_definitions(kael PRIVATE KAEL_NO_CACHING_ALLOCATOR)
endif ()

if (KAEL_ENABLE_CUDA)
    target_link_libraries(kael PUBLIC CUDA::cudart)
    if (KAEL_USE_CUBLAS)
        target_link_libraries(kael PRIVATE CUDA::cublas)
        target_compile_definitions(kael PRIVATE KAEL_USE_CUBLAS)
    endif ()
endif ()

find_package(Threads REQUIRED)
target_link_libraries(kael PUBLIC Threads::Threads)

if (KAEL_USE_BLAS)
    if (APPLE)
        find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
        target_link_libraries(kael PRIVATE ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(kael PRIVATE KAEL_USE_ACCELERATE ACCELERATE_NEW_LAPACK)
    else ()
        find_package(BLAS REQUIRED)
        target_link_libraries(kael PRIVATE BLAS::BLAS)
        target_compile_definitions(kael PRIVATE KAEL_USE_CBLAS)
    endif ()
endif ()

if (KAEL_NATIVE)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-march=native KAEL_HAS_MARCH_NATIVE)
    check_cxx_compiler_flag(-mcpu=native KAEL_HAS_MCPU_NATIVE)
    if (KAEL_HAS_MARCH_NATIVE)
        target_compile_options(kael PRIVATE -march=native)
    elseif (KAEL_HAS_MCPU_NATIVE)
        target_compile_options(kael PRIVATE -mcpu=native)
    endif ()
endif ()

include(FetchContent)

if (KAEL_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    )
    FetchContent_MakeAvailable(googletest)
    enable_testing()
    add_subdirectory(tests)
endif ()

if (KAEL_BUILD_PYTHON)
    add_subdirectory(python)
endif ()

if (KAEL_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

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
