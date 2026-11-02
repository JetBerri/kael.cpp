add_executable(kael_bench
    bench_elementwise.cpp
    bench_matmul.cpp
    bench_compare.cpp
)

target_link_libraries(kael_bench PRIVATE kael::kael benchmark::benchmark_main)
kael_configure_target(kael_bench)
if (KAEL_USE_BLAS)
    target_compile_definitions(kael_bench PRIVATE KAEL_BENCH_BLAS)
endif ()
