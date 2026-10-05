add_executable(kael_tests
    test_shape.cpp
)

target_link_libraries(kael_tests PRIVATE kael::kael GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(kael_tests)