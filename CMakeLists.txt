cmake_minimum_required(VERSION 3.14)
project(stam_fpga VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_BUILD_TYPE Debug)

file(GLOB SOURCES "src/*.cpp")
add_executable (exe ${SOURCES})
target_include_directories (exe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

include(${CMAKE_BINARY_DIR}/conan_paths.cmake)

find_package(HDF5)
if(TARGET HDF5::HDF5)
   target_link_libraries(exe HDF5::HDF5)
endif()
