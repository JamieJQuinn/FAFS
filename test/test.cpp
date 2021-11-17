#include <iostream>
#include <catch2/catch_test_macros.hpp>
#include <ocl_utility.hpp>
#include <opencl_kernels.hpp>
#include <array2d.hpp>

TEST_CASE( "Test filling array with value", "[array, ocl]" ) {
  const int nx = 16;
  const int ny = 16;
  const int ng = 0;

  Array arr(nx, ny, ng);

  arr.initOnDevice();

  auto program = buildProgramFromString(FAFS_PROGRAM);
  auto fill = cl::KernelFunctor<cl::Buffer, real, int, int, int>(program, "fill");

  fill(arr.range, arr.getDeviceData(), 1.0f, arr.nx, arr.ny, arr.ng);

  arr.toHost();

  for(int i=0; i<nx; ++i) {
    for(int j=0; j<nx; ++j) {
      assert(arr(i,j) == 1.0f);
    }
  }
}
