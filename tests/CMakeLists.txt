add_executable(kael_tests
    test_shape.cpp
    test_dtype.cpp
    test_storage.cpp
    test_tensor_impl.cpp
)

target_link_libraries(kael_tests PRIVATE kael::kael GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(kael_tests)
kael_configure_target(kael_tests)
