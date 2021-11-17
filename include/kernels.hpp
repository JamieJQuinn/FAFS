#pragma once

#include <ocl_utility.hpp>

class Kernels {
  public:
    Kernels();
  private:
    cl::Program program;
    cl::KernelFunctor<cl::Buffer, int, int, int> addOne;
};
