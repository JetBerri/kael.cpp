add_executable(kael_tests
    test_shape.cpp
    test_dtype.cpp
    test_storage.cpp
    test_tensor_impl.cpp
    test_tensor.cpp
    test_factory.cpp
    test_random.cpp
    test_view.cpp
    test_iterator.cpp
    test_kernels.cpp
    test_golden_support.cpp
    test_golden.cpp
    test_ops_binary.cpp
    test_ops_unary.cpp
    test_ops_inplace.cpp
    test_ops_reduce.cpp
    test_ops_linalg.cpp
    test_ops_indexing.cpp
    test_ops_nn.cpp
    test_autograd_grad_mode.cpp
    test_autograd_fields.cpp
    test_autograd_engine.cpp
    test_autograd_gradcheck.cpp
    test_autograd_sum_to_shape.cpp
    test_autograd_mlp.cpp
    test_nn_module.cpp
    test_nn_layers.cpp
    test_optim.cpp
    test_data.cpp
    test_training.cpp
    test_parallel.cpp
    test_matmul_versions.cpp
    test_parallel_ops.cpp
    test_allocator.cpp
    test_errors.cpp
    support/npy.cpp
)

target_link_libraries(kael_tests PRIVATE kael::kael GTest::gtest_main)
target_include_directories(kael_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(kael_tests PRIVATE
    KAEL_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden"
    KAEL_EXPECTED_VERSION="${PROJECT_VERSION}"
)

include(GoogleTest)
gtest_discover_tests(kael_tests)
kael_configure_target(kael_tests)

add_executable(micrograd ${PROJECT_SOURCE_DIR}/experiments/micrograd.cpp)
target_compile_features(micrograd PRIVATE cxx_std_20)
kael_configure_target(micrograd)
add_test(NAME micrograd COMMAND micrograd)
