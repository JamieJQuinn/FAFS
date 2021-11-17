#include <iostream>
#include <memory>
#include <cassert>

#include <ocl_utility.hpp>
#include <openmp_implementation.hpp>

int main() {
  runCPU();

  //int error = setDefaultPlatform("CUDA");
  //if (error < 0) return -1;

  //cl::DeviceCommandQueue deviceQueue = cl::DeviceCommandQueue::makeDefault(
      //cl::Context::getDefault(), cl::Device::getDefault());

  //cl::Buffer d_A = cl::Buffer(h_A.begin(), h_A.end(), true);
  //cl::Buffer d_B = cl::Buffer(h_B.begin(), h_B.end(), true);
  //cl::Buffer d_C = cl::Buffer(CL_MEM_READ_WRITE, sizeof(float)*h_C.size());

  //// Allocate temp space local to workgroups
  //cl::LocalSpaceArg d_wrk = cl::Local(sizeof(float)*N);

  //cl::Program program = buildProgram("mat_mult.cl");
  //// Build kernel with extra workgroup space
  //auto mat_mult_cl = cl::KernelFunctor<
    //int, cl::Buffer, cl::Buffer, cl::Buffer, cl::LocalSpaceArg
    //>(program, kernelName);

  //auto start = high_resolution_clock::now();

  //// Pass in workgroup
  //mat_mult_cl(cl::EnqueueArgs(cl::NDRange(c.ng, c.ng), cl::NDRange(N), cl::NDRange(N/4)), N, d_A, d_B, d_C, d_wrk);

  //cl::copy(d_C, h_C.begin(), h_C.end());
}
