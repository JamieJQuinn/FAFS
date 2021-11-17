#include <kernels.hpp>
#include <opencl_kernels.hpp>

Kernels::Kernels() :
  program{buildProgramFromString(FAFS_PROGRAM)},
  fill(program, "fill")
{}
