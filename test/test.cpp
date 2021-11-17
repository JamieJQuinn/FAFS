#include <catch2/catch_test_macros.hpp>
#include <ocl_utility.hpp>
#include <kernels.hpp>

//TEST_CASE( "Simple OpenCL example works", "[array, ocl]" ) {
  //cl::DeviceCommandQueue deviceQueue = cl::DeviceCommandQueue::makeDefault(
      //cl::Context::getDefault(), cl::Device::getDefault());
  //const int nx = 16;
  //const int ny = 16;
  //const int ng = 0;

  //Array arr(nx, ny, ng);

  //arr.initOnDevice(false);

  //REQUIRE( Factorial(1) == 1 );
  //REQUIRE( Factorial(2) == 2 );
  //REQUIRE( Factorial(3) == 6 );
  //REQUIRE( Factorial(10) == 3628800 );
//}

TEST_CASE( "Loading kernels works", "[kernels]") {
  Kernels kernels;
}
