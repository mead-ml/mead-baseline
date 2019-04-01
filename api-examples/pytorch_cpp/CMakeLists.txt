cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(tag-text tag-text.cpp json.hpp)
target_link_libraries(tag-text "${TORCH_LIBRARIES}")
set_property(TARGET tag-text PROPERTY CXX_STANDARD 11)

add_executable(classify-text classify-text.cpp json.hpp)
target_link_libraries(classify-text "${TORCH_LIBRARIES}")
set_property(TARGET classify-text PROPERTY CXX_STANDARD 11)
