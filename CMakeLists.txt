cmake_minimum_required(VERSION 3.14)
project(stam_fpga VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_BUILD_TYPE Release)

file(GLOB SOURCES "src/*.cpp")
add_executable (exe ${SOURCES})
target_include_directories (exe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
