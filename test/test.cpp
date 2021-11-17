#include <iostream>
#include <catch2/catch_test_macros.hpp>
#include <ocl_utility.hpp>
#include <opencl_kernels.hpp>
#include <array2d.hpp>

TEST_CASE( "Test filling array with value", "[array, ocl]" ) {
  const int nx = 16;
  const int ny = 16;
  const int ng = 1;

  Array arr(nx, ny, ng);

  arr.initOnDevice();

  auto program = buildProgramFromString(FAFS_PROGRAM);
  auto fill = fillKernelType(program, "fill");

  fill(arr.range, arr.getDeviceData(), 1.0f, arr.nx, arr.ny, arr.ng);

  arr.toHost();

  for(int i=0; i<nx; ++i) {
    for(int j=0; j<nx; ++j) {
      assert(arr(i,j) == 1.0f);
    }
  }

  for(int i=0; i<nx; ++i) {
    assert(arr(i,-1) == 0.0f);
  }

  for(int j=0; j<nx; ++j) {
    assert(arr(-1,j) == 0.0f);
  }
}
