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
    support/npy.cpp
)

target_link_libraries(kael_tests PRIVATE kael::kael GTest::gtest_main)
target_include_directories(kael_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(kael_tests PRIVATE KAEL_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden")

include(GoogleTest)
gtest_discover_tests(kael_tests)
kael_configure_target(kael_tests)
