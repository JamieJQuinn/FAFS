#pragma once

#include <iostream>

#include <ocl_utility.hpp>
#include <opencl_kernels.hpp>

template<class T>
T createKernelFunctor(const cl::Program& program, const std::string& kernelName) {
  cl::Kernel kernel;
  try {
    kernel = cl::Kernel(program, kernelName.c_str());
  } catch (cl::Error& e) {
    std::cout << e.what() << std::endl;
  }
  return T(kernel);
}

class Kernels {
  public:
    Kernels();
    cl::Program program;
    fillKernel fill;
    vonNeumannKernel applyVonNeumannBC_x;
    vonNeumannKernel applyVonNeumannBC_y;
    advanceEulerKernel advanceEuler;
};
