add_executable(kael_bench
    bench_elementwise.cpp
)

target_link_libraries(kael_bench PRIVATE kael::kael benchmark::benchmark_main)
kael_configure_target(kael_bench)
