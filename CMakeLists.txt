cmake_minimum_required(VERSION 3.24)
project(kael VERSION 0.1.0 LANGUAGES CXX)

add_library(kael
    src/shape.cpp
)

add_library(kael::kael ALIAS kael)

target_include_directories(kael PUBLIC include)
target_compile_features(kael PUBLIC cxx_std_20)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_subdirectory(tests)