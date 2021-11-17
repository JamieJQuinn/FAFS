#include <iostream>
#include <memory>
#include <cassert>

#include <ocl_utility.hpp>
#include <constants.hpp>
#include <precision.hpp>
#include <openmp_implementation.hpp>
#include <variables.hpp>
#include <array2d.hpp>
#include <opencl_kernels.hpp>

int runOCL() {
  const Constants c;

  int error = setDefaultPlatform("CUDA");
  if (error < 0) return -1;

  cl::DeviceCommandQueue deviceQueue = cl::DeviceCommandQueue::makeDefault(
      cl::Context::getDefault(), cl::Device::getDefault());

  Variables vars(c);

  setInitialConditions(vars);

  cl::Buffer d_vx = cl::Buffer(vars.vx.begin(), vars.vx.end(), false);

  auto &var = vars.vx;
  auto range = cl::EnqueueArgs(cl::NDRange(var.ng, var.ng), cl::NDRange(var.nx, var.ny), cl::NDRange(var.nx, var.ny));

  addOne_cl(range, var.getDeviceData(), vars.vx.nx, vars.vx.ny, vars.vx.ng);

  cl::copy(d_vx, vars.vx.begin(), vars.vx.end());

  for(int i=0; i<vars.vx.nx; ++i) {
    for(int j=0; j<vars.vx.nx; ++j) {
      //std::cout << vars.vx(i,j) << ", ";
      assert(vars.vx(i,j) == 1.0f);
    }
  }

  return 0;
}

int main() {
  return runOCL();
}
