#include <iostream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <ocl_utility.hpp>
#include <ocl_array.hpp>
#include <kernels.hpp>
#include <user_kernels.hpp>
#include <hdffile.hpp>

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
  applyVonNeumannBC(arr);

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

  real alpha = 1.0f/(1.0f + 2.0f*dt/Re*(1.0f/(dx*dx) + 1.0f/(dy*dy)));
  real beta  = -Re*dx*dx/dt;
  real gamma = -Re*dy*dy/dt;
  runJacobiIteration(resImplicit, temp1, temp2, alpha, beta, gamma, resImplicit);

  resImplicit.toHost();

  OpenCLArray resExplicit(nx, ny, ng);

  resExplicit.fill(1.0f, true);
  temp1.fill(0.0f, true);

  calcDiffusionTerm(temp1, resExplicit, dx, dy, Re);
  advanceEuler(resExplicit, temp1, dt);

  resExplicit.toHost();

  for(int i=0; i<nx; ++i) {
    for(int j=0; j<nx; ++j) {
      std::cout << i << ", " << j << std::endl;
      REQUIRE(resImplicit(i,j) == Catch::Approx(resExplicit(i,j)));
    }
  }
}

TEST_CASE( "Test saving OpenCLArray", "[ocl]") {
  const int nx = 64;
  const int ny = 64;
  const int ng = 1;

  OpenCLArray arr(nx, ny, ng, "test");

  for(int i=0; i< arr.nx; ++i) {
    for(int j=0; j< arr.ny; ++j) {
      arr(i,j) = i+2.0*j;
    }
  }

  // Write file
  HDFFile file("test.hdf5", false);
  arr.saveTo(file.file);
  file.close();

  // Read in a new array
  OpenCLArray arr_in(nx, ny, ng, "test");
  file.open("test.hdf5");
  arr_in.load(file.file);
  file.close();

  // Check array is the same as the one we wrote
  for(int i=0; i< arr.nx; ++i) {
    for(int j=0; j< arr.ny; ++j) {
      arr_in(i,j) = arr(i,j);
    }
  }
}
