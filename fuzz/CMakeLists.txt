foreach (target fuzz_safetensors fuzz_idx)
    add_executable(${target} ${target}.cpp)
    target_link_libraries(${target} PRIVATE kael::kael)
    target_compile_options(${target} PRIVATE -fsanitize=fuzzer)
    target_link_options(${target} PRIVATE -fsanitize=fuzzer)
    kael_configure_target(${target})
endforeach ()

target_compile_options(kael PRIVATE -fsanitize=fuzzer-no-link)
