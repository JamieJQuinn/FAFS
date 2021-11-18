#include <iostream>
#include <catch2/catch_test_macros.hpp>
#include <ocl_utility.hpp>
#include <ocl_array.hpp>
#include <kernels.hpp>

TEST_CASE( "Test filling array with value", "[ocl]" ) {
  const int nx = 16;
  const int ny = 16;
  const int ng = 1;

  openCLArray arr(nx, ny, ng);

  arr.initOnDevice();

  arr.fill(1.0f);

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

  openCLArray arr(nx, ny, ng);

  arr.initOnDevice();

  arr.fill(1.0f);
  arr.setLowerBoundary(2.0f);
  arr.setUpperBoundary(3.0f);
  arr.setLeftBoundary(4.0f);
  arr.setRightBoundary(5.0f);
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

  openCLArray arr(nx, ny, ng);

  arr.initOnDevice();

  arr.fill(1.0f);
  g_kernels.applyVonNeumannBC_y(arr.lowerBound, arr.getDeviceData(), arr.nx, arr.ny, arr.ng);
  g_kernels.applyVonNeumannBC_x(arr.leftBound, arr.getDeviceData(), arr.nx, arr.ny, arr.ng);

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

  openCLArray arr(nx, ny, ng);
  openCLArray ddt(nx, ny, ng);

  arr.initOnDevice();
  ddt.initOnDevice();

  arr.fill(1.0f);
  ddt.fill(1.0f);
  g_kernels.advanceEuler(arr.interior, arr.getDeviceData(), ddt.getDeviceData(), dt, arr.nx, arr.ny, arr.ng);

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

TEST_CASE( "Test calculating diffusion term", "[ocl") {
  const int nx = 16;
  const int ny = 16;
  const int ng = 1;

  openCLArray arr(nx, ny, ng);
  openCLArray res(nx, ny, ng);

  arr.initOnDevice();
  res.initOnDevice();

  arr.fill(1.0f, true);
  g_kernels.calcDiffusionTerm(res.interior, res.getDeviceData(), arr.getDeviceData(), 1.0f, 1.0f, res.nx, res.ny, res.ng);

  res.toHost();

  for(int i=0; i<nx; ++i) {
    for(int j=0; j<nx; ++j) {
      REQUIRE(res(i,j) == 0.0f);
    }
  }
}

TEST_CASE( "Test calculating advection term", "[ocl") {
  const int nx = 16;
  const int ny = 16;
  const int ng = 1;

  openCLArray arr(nx, ny, ng);
  openCLArray vx(nx, ny, ng);
  openCLArray vy(nx, ny, ng);
  openCLArray res(nx, ny, ng);

  arr.initOnDevice();
  vx.initOnDevice();
  vy.initOnDevice();
  res.initOnDevice();

  arr.fill(1.0f, true);
  vx.fill(1.0f, true);
  vy.fill(1.0f, true);
  g_kernels.calcAdvectionTerm(res.interior, res.getDeviceData(), arr.getDeviceData(), vx.getDeviceData(), vy.getDeviceData(), 1.0f, 1.0f, res.nx, res.ny, res.ng);

  res.toHost();

  for(int i=0; i<nx; ++i) {
    for(int j=0; j<nx; ++j) {
      REQUIRE(res(i,j) == 0.0f);
    }
  }
}
