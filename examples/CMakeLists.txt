add_executable(mnist_mlp mnist_mlp.cpp)
target_link_libraries(mnist_mlp PRIVATE kael::kael)
kael_configure_target(mnist_mlp)

add_executable(char_gpt char_gpt.cpp)
target_link_libraries(char_gpt PRIVATE kael::kael)
kael_configure_target(char_gpt)

add_executable(gpt2_generate gpt2_generate.cpp)
target_link_libraries(gpt2_generate PRIVATE kael::kael)
kael_configure_target(gpt2_generate)
