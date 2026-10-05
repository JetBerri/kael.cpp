add_test([=[ShapeString.NormalCase]=]  /Users/diqueran/Documents/Proyectos/kael.cpp/build/tests/kael_tests [==[--gtest_filter=ShapeString.NormalCase]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[ShapeString.NormalCase]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/diqueran/Documents/Proyectos/kael.cpp/tests/test_shape.cpp:4]==]
    WORKING_DIRECTORY [==[/Users/diqueran/Documents/Proyectos/kael.cpp/build/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[ShapeString.EmptyCase]=]  /Users/diqueran/Documents/Proyectos/kael.cpp/build/tests/kael_tests [==[--gtest_filter=ShapeString.EmptyCase]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[ShapeString.EmptyCase]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/diqueran/Documents/Proyectos/kael.cpp/tests/test_shape.cpp:12]==]
    WORKING_DIRECTORY [==[/Users/diqueran/Documents/Proyectos/kael.cpp/build/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[ShapeString.OneDimension]=]  /Users/diqueran/Documents/Proyectos/kael.cpp/build/tests/kael_tests [==[--gtest_filter=ShapeString.OneDimension]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[ShapeString.OneDimension]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/diqueran/Documents/Proyectos/kael.cpp/tests/test_shape.cpp:20]==]
    WORKING_DIRECTORY [==[/Users/diqueran/Documents/Proyectos/kael.cpp/build/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[ShapeString.ZeroDimension]=]  /Users/diqueran/Documents/Proyectos/kael.cpp/build/tests/kael_tests [==[--gtest_filter=ShapeString.ZeroDimension]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[ShapeString.ZeroDimension]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/diqueran/Documents/Proyectos/kael.cpp/tests/test_shape.cpp:28]==]
    WORKING_DIRECTORY [==[/Users/diqueran/Documents/Proyectos/kael.cpp/build/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[ShapeString.BigDimension]=]  /Users/diqueran/Documents/Proyectos/kael.cpp/build/tests/kael_tests [==[--gtest_filter=ShapeString.BigDimension]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[ShapeString.BigDimension]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/diqueran/Documents/Proyectos/kael.cpp/tests/test_shape.cpp:36]==]
    WORKING_DIRECTORY [==[/Users/diqueran/Documents/Proyectos/kael.cpp/build/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
set(kael_tests_TESTS [==[ShapeString.NormalCase]==] [==[ShapeString.EmptyCase]==] [==[ShapeString.OneDimension]==] [==[ShapeString.ZeroDimension]==] [==[ShapeString.BigDimension]==])
