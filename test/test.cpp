#include <iostream>
#include <catch2/catch_test_macros.hpp>
#include <ocl_utility.hpp>
#include <opencl_kernels.hpp>
#include <array2d.hpp>

TEST_CASE( "Test filling array with value", "[ocl]" ) {
  const int nx = 16;
  const int ny = 16;
  const int ng = 1;

  Array arr(nx, ny, ng);

  arr.initOnDevice();

  auto program = buildProgramFromString(FAFS_PROGRAM);
  auto fill = fillKernel(program, "fill");

  fill(arr.range, arr.getDeviceData(), 1.0f, arr.nx, arr.ny, arr.ng);

  arr.toHost();

  // Ensure interior has been filled
  for(int i=0; i<nx; ++i) {
    for(int j=0; j<nx; ++j) {
      REQUIRE(arr(i,j) == 1.0f);
    }
  }

  // Ensure boundaries aren't affected
  for(int i=0; i<nx; ++i) {
    REQUIRE(arr(i,-1) == 0.0f);
    REQUIRE(arr(i,ny) == 0.0f);
  }

  for(int j=0; j<nx; ++j) {
    REQUIRE(arr(-1,j) == 0.0f);
    REQUIRE(arr(nx,j) == 0.0f);
  }

}

TEST_CASE( "Test applying Dirichlet boundary conditions on device", "[boundary, ocl]" ) {
  const int nx = 16;
  const int ny = 16;
  const int ng = 1;

  Array arr(nx, ny, ng);

  arr.initOnDevice();

  auto program = buildProgramFromString(FAFS_PROGRAM);
  auto fill = fillKernel(program, "fill");

  fill(arr.range, arr.getDeviceData(), 1.0f, arr.nx, arr.ny, arr.ng);
  fill(arr.lowerBRange, arr.getDeviceData(), 2.0f, arr.nx, arr.ny, arr.ng);
  fill(arr.upperBRange, arr.getDeviceData(), 3.0f, arr.nx, arr.ny, arr.ng);
  fill(arr.leftBRange, arr.getDeviceData(), 4.0f, arr.nx, arr.ny, arr.ng);
  fill(arr.rightBRange, arr.getDeviceData(), 5.0f, arr.nx, arr.ny, arr.ng);

  arr.toHost();

  for(int i=0; i<nx; ++i) {
    for(int j=0; j<nx; ++j) {
      REQUIRE(arr(i,j) == 1.0f);
    }
  }

  for(int i=0; i<nx; ++i) {
    REQUIRE(arr(i,-1) == 2.0f);
  }

  for(int i=0; i<nx; ++i) {
    REQUIRE(arr(i,ny) == 3.0f);
  }

  for(int j=0; j<ny; ++j) {
    REQUIRE(arr(-1,j) == 4.0f);
  }

  for(int j=0; j<ny; ++j) {
    REQUIRE(arr(nx,j) == 5.0f);
  }
}

TEST_CASE( "Test applying von Neumann boundary conditions on device", "[boundary, ocl]" ) {
  const int nx = 16;
  const int ny = 16;
  const int ng = 1;

  Array arr(nx, ny, ng);

  arr.initOnDevice();

  auto program = buildProgramFromString(FAFS_PROGRAM);
  auto fill = fillKernel(program, "fill");
  auto applyVonNeumannBC_y = vonNeumannKernel(program, "applyVonNeumannBC_y");
  auto applyVonNeumannBC_x = vonNeumannKernel(program, "applyVonNeumannBC_x");

  fill(arr.range, arr.getDeviceData(), 1.0f, arr.nx, arr.ny, arr.ng);
  applyVonNeumannBC_y(arr.lowerBRange, arr.getDeviceData(), arr.nx, arr.ny, arr.ng);
  applyVonNeumannBC_x(arr.leftBRange, arr.getDeviceData(), arr.nx, arr.ny, arr.ng);

  arr.toHost();

  for(int i=0; i<nx; ++i) {
    for(int j=0; j<nx; ++j) {
      REQUIRE(arr(i,j) == 1.0f);
    }
  }

  for(int i=0; i<nx; ++i) {
    REQUIRE(arr(i,-1) == 1.0f);
  }

  for(int i=0; i<nx; ++i) {
    REQUIRE(arr(i,ny) == 1.0f);
  }

  for(int j=0; j<ny; ++j) {
    REQUIRE(arr(-1,j) == 1.0f);
  }

  for(int j=0; j<ny; ++j) {
    REQUIRE(arr(nx,j) == 1.0f);
  }
}

TEST_CASE( "Test Euler method", "[ocl]" ) {
  const int nx = 16;
  const int ny = 16;
  const int ng = 1;
  const real dt = 1.0f;

  Array arr(nx, ny, ng);
  Array ddt(nx, ny, ng);

  arr.initOnDevice();
  ddt.initOnDevice();

  auto program = buildProgramFromString(FAFS_PROGRAM);
  auto fill = fillKernel(program, "fill");
  auto advanceEuler = advanceEulerKernel(program, "advanceEuler");

  fill(arr.range, arr.getDeviceData(), 1.0f, arr.nx, arr.ny, arr.ng);
  fill(ddt.range, ddt.getDeviceData(), 1.0f, ddt.nx, ddt.ny, ddt.ng);
  advanceEuler(arr.range, arr.getDeviceData(), ddt.getDeviceData(), dt, arr.nx, arr.ny, arr.ng);

  arr.toHost();

  for(int i=0; i<nx; ++i) {
    for(int j=0; j<nx; ++j) {
      REQUIRE(arr(i,j) == 2.0f);
    }
  }

  for(int i=0; i<nx; ++i) {
    REQUIRE(arr(i,-1) == 0.0f);
    REQUIRE(arr(i,ny) == 0.0f);
  }

  for(int j=0; j<nx; ++j) {
    REQUIRE(arr(-1,j) == 0.0f);
    REQUIRE(arr(nx,j) == 0.0f);
  }
}
