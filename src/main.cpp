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

  cl::Program program = buildProgramFromString(FAFS_PROGRAM);
  int cl_error;
  std::string kernelName = "add_one";

  auto addOne_cl = cl::KernelFunctor<
    cl::Buffer, int, int, int
    >(program, kernelName, &cl_error);

  if(cl_error != 0) {
    std::string errorMsg = std::string("OpenCL: Could not create kernel ") + std::string(kernelName) + ": " + std::to_string(cl_error);
    throw std::runtime_error(errorMsg);
  }

  addOne_cl(cl::EnqueueArgs(cl::NDRange(vars.vx.ng, vars.vx.ng), cl::NDRange(vars.vx.nx, vars.vx.ny), cl::NDRange(vars.vx.nx, vars.vx.ny)), d_vx, vars.vx.nx, vars.vx.ny, vars.vx.ng);

  cl::copy(d_vx, vars.vx.begin(), vars.vx.end());

  for(int i=-1; i<=vars.vx.nx; ++i) {
    for(int j=-1; j<=vars.vx.nx; ++j) {
      std::cout << vars.vx(i,j) << ", ";
      //assert(vars.vx(i,j) == 1.0f);
    }
    std::cout << std::endl;
  }

  return 0;
}

int main() {
  return runOCL();
}
