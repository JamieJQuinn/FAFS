#pragma once

#include <iostream>

#include <ocl_utility.hpp>
#include <precision.hpp>

// Procedure for adding a kernel:
// 1. add opencl kernel code to FAFS_PROGRAM in src/kernels.cpp
// 2. typedef kernel below
// 3. add kernel object to Kernels class definition below
// 4. add kernel construction to Kernels constructor in src/kernels.cpp

typedef cl::KernelFunctor<cl::Buffer, real, int, int, int> fillKernel;
typedef cl::KernelFunctor<cl::Buffer, cl::Buffer, real, int, int, int> advanceEulerKernel;
typedef cl::KernelFunctor<cl::Buffer, cl::Buffer, real, real, real, int, int, int> calcDiffusionKernel;
typedef cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, real, real, int, int, int> calcAdvectionKernel;
typedef cl::KernelFunctor<cl::Buffer, int, int, int> vonNeumannKernel;
typedef cl::KernelFunctor<cl::Buffer, cl::Buffer, real, real, real, cl::Buffer, int, int, int> applyJacobiKernel;
typedef cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, real, real, int, int, int> calcDivergence_k;
typedef cl::KernelFunctor<cl::Buffer, cl::Buffer, real, int, int, int> applyProjection_k;
typedef cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, real, real, real, int, int, int> advect_k;

class Kernels {
  public:
    Kernels();
  protected:
    cl::Program program; // This must be initialised before kernels
  public:
    fillKernel fill;
    vonNeumannKernel applyVonNeumannBC_x;
    vonNeumannKernel applyVonNeumannBC_y;
    advanceEulerKernel advanceEuler;
    calcDiffusionKernel calcDiffusionTerm;
    calcAdvectionKernel calcAdvectionTerm;
    applyJacobiKernel applyJacobiStep;
    calcDivergence_k calcDivergence;
    applyProjection_k applyProjectionX;
    applyProjection_k applyProjectionY;
    advect_k advect;
};

extern const std::string FAFS_PROGRAM;

extern Kernels g_kernels;

template<class T>
T createKernelFunctor(const cl::Program& program, const std::string& kernelName) {
  cl::Kernel kernel;
  try {
    kernel = cl::Kernel(program, kernelName.c_str());
  } catch (cl::Error& e) {
    std::cout << e.what() << ", " << e.err() << std::endl;
  }
  return T(kernel);
}
