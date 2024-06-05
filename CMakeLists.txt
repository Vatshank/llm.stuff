cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(llm.stuff)

find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)
 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")

link_libraries("${TORCH_LIBRARIES}" ${TORCH_PYTHON_LIBRARY})

pybind11_add_module(cpp_funcs pt_binding.cpp kernels.cu)

set_property(GLOBAL PROPERTY CXX_STANDARD 17)
