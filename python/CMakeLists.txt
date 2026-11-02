find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_C NB_STATIC
    src/bindings.cpp
    src/tensor_bind.cpp
    src/ops_bind.cpp
    src/autograd_bind.cpp
    src/interop_bind.cpp
    src/models_bind.cpp
)
target_link_libraries(_C PRIVATE kael::kael)
target_compile_features(_C PRIVATE cxx_std_20)
install(TARGETS _C LIBRARY DESTINATION kael)

nanobind_add_stub(_C_stub
    MODULE _C
    OUTPUT _C.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:_C>
    DEPENDS _C
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_C.pyi DESTINATION kael)
