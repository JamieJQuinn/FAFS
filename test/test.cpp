#include <iostream>
#include <catch2/catch_test_macros.hpp>
#include <ocl_utility.hpp>
#include <ocl_array.hpp>
#include <kernels.hpp>
#include <user_kernels.hpp>

TEST_CASE( "Test filling array with value", "[ocl]" ) {
  const int nx = 64;
  const int ny = 64;
  const int ng = 1;

  OpenCLArray arr(nx, ny, ng);

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
  const int nx = 64;
  const int ny = 64;
  const int ng = 1;

  OpenCLArray arr(nx, ny, ng);

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

  OpenCLArray arr(nx, ny, ng);

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

  OpenCLArray arr(nx, ny, ng);
  OpenCLArray ddt(nx, ny, ng);

  arr.fill(1.0f);
  ddt.fill(1.0f);
  advanceEuler(arr, ddt, dt);

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

TEST_CASE( "Test calculating diffusion term", "[ocl]") {
  const int nx = 16;
  const int ny = 16;
  const int ng = 1;

  OpenCLArray arr(nx, ny, ng);
  OpenCLArray res(nx, ny, ng);

  arr.fill(1.0f, true);
  calcDiffusionTerm(res, arr, 1.0f, 1.0f, 1.0f);

  res.toHost();

  for(int i=0; i<nx; ++i) {
    for(int j=0; j<nx; ++j) {
      REQUIRE(res(i,j) == 0.0f);
    }
  }
}

TEST_CASE( "Test calculating advection term", "[ocl]") {
  const int nx = 16;
  const int ny = 16;
  const int ng = 1;

  OpenCLArray arr(nx, ny, ng);
  OpenCLArray vx(nx, ny, ng);
  OpenCLArray vy(nx, ny, ng);
  OpenCLArray res(nx, ny, ng);

  arr.fill(1.0f, true);
  vx.fill(1.0f, true);
  vy.fill(1.0f, true);
  calcAdvectionTerm(res, arr, vx, vy, 1.0f, 1.0f);

  res.toHost();

  for(int i=0; i<nx; ++i) {
    for(int j=0; j<nx; ++j) {
      REQUIRE(res(i,j) == 0.0f);
    }
  }
}

TEST_CASE( "Test Jacobi iteration", "[ocl]") {
  const int nx = 16;
  const int ny = 16;
  const int ng = 1;

  const real dx = 1.0f/(nx-1);
  const real dy = 1.0f/(ny-1);
  const real dt = 0.01f;
  const real Re = 10.0f;

  OpenCLArray resImplicit(nx, ny, ng);
  OpenCLArray temp1(nx, ny, ng);
  OpenCLArray temp2(nx, ny, ng);

  resImplicit.fill(1.0f, true);
  temp1.fill(1.0f, true);
  temp2.fill(1.0f, true);

  const real alpha = Re*dx*dy/dt;
  const real beta = 4.0f+alpha;
  runJacobiIteration(resImplicit, temp1, temp2, alpha, beta, resImplicit);

  resImplicit.toHost();

  OpenCLArray resExplicit(nx, ny, ng);

  resExplicit.fill(1.0f, true);
  temp1.fill(0.0f, true);

  calcDiffusionTerm(temp1, resExplicit, dx, dy, Re);
  advanceEuler(resExplicit, temp1, dt);

  resExplicit.toHost();

  for(int i=0; i<nx; ++i) {
    for(int j=0; j<nx; ++j) {
      REQUIRE(resImplicit(i,j) == resExplicit(i,j));
    }
  }
}
