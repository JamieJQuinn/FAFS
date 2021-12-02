#include <iostream>
#include <cmath>

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

TEST_CASE( "Test Jacobi iteration on Poisson eqn", "[ocl]") {
  const int nx = 32;
  const int ny = nx;
  const int ng = 1;

  const real dx = 1.0f/(nx+2*ng-1);
  const real dy = 1.0f/(ny+2*ng-1);
  const real dt = 0.01f;
  const real Re = 10.0f;

  OpenCLArray result(nx, ny, ng);
  OpenCLArray b(nx, ny, ng);

  for(int i=-ng; i<b.nx+ng; ++i) {
    for(int j=-ng; j<b.ny+ng; ++j) {
      real x = 0 + (i+ng)*dx;
      real y = 0 + (j+ng)*dy;
      b(i,j) = cos(M_PI*x)*sin(M_PI*y);
    }
  }

  b.toDevice();

  OpenCLArray temp1(nx, ny, ng);
  OpenCLArray temp2(nx, ny, ng);

  temp1.fill(0.0f, true);
  temp2.fill(0.0f, true);

  real alpha = -0.5f*(dx*dx*dy*dy)/(dx*dx + dy*dy);
  real beta = dx*dx;
  real gamma = dy*dy;

  for(int i=0; i<100; ++i) {
    applyVonNeumannBC(temp1);
    g_kernels.applyJacobiStep(temp2.interior, temp2.getDeviceData(), temp1.getDeviceData(), alpha, beta, gamma, b.getDeviceData(), temp2.nx, temp2.ny, temp2.ng);
    temp1.swapData(temp2);
  }

  temp1.toHost();

  for(int i=0; i<nx; ++i) {
    for(int j=0; j<ny; ++j) {
      temp2(i,j) = (temp1(i+1,j) + temp1(i-1,j) + temp1(i, j+1) + temp1(i,j-1) - 4.0*temp1(i,j))/(dx*dx);
    }
  }

  for(int i=0; i<nx; ++i) {
    for(int j=0; j<ny; ++j) {
      std::cout << i << ", " << j << std::endl;
      REQUIRE(temp2(i,j) == Catch::Approx(b(i,j)).margin(0.001));
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
