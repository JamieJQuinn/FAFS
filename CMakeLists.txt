cmake_minimum_required(VERSION 3.14)
project(fafs VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_BUILD_TYPE Debug)

file(GLOB SOURCES "src/*.cpp")
add_executable (exe ${SOURCES})
target_include_directories (exe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(exe PRIVATE -Wall -Wextra)

include(${CMAKE_BINARY_DIR}/conan_paths.cmake)

find_package(HDF5)
if(TARGET HDF5::HDF5)
  target_link_libraries(exe PUBLIC HDF5::HDF5)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(exe PUBLIC OpenMP::OpenMP_CXX)
endif()

find_package(OpenCL REQUIRED)
if(TARGET OpenCL::OpenCL)
  target_link_libraries(exe PUBLIC OpenCL::OpenCL)
endif()

Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.0.0-preview3
)

FetchContent_MakeAvailable(Catch2)

file(GLOB TEST_SOURCES "test/*.cpp")
add_executable (tests ${TEST_SOURCES})
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
