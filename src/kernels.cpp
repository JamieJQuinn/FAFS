#include <kernels.hpp>
#include <opencl_kernels.hpp>

Kernels::Kernels() :
  program{buildProgramFromString(FAFS_PROGRAM)},
  addOne(program, "add_one")
{}
