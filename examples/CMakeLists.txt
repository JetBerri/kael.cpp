add_executable(mnist_mlp mnist_mlp.cpp)
target_link_libraries(mnist_mlp PRIVATE kael::kael)
kael_configure_target(mnist_mlp)
