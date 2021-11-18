#pragma once

#include <precision.hpp>
#include <ocl_utility.hpp>

typedef cl::KernelFunctor<cl::Buffer, real, int, int, int> fillKernel;
typedef cl::KernelFunctor<cl::Buffer, cl::Buffer, real, int, int, int> advanceEulerKernel;
typedef cl::KernelFunctor<cl::Buffer, int, int, int> vonNeumannKernel;

extern std::string FAFS_PROGRAM;
