cmake_minimum_required(VERSION 3.16)
project(radar_engine LANGUAGES CXX)

find_package(pybind11 REQUIRED)

add_library(radar_engine MODULE src/main.cpp)
target_link_libraries(radar_engine PRIVATE pybind11::module)
target_compile_features(radar_engine PRIVATE cxx_std_17)

set_target_properties(
    radar_engine
    PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
