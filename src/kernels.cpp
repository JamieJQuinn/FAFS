#include <kernels.hpp>
#include <opencl_kernels.hpp>

Kernels::Kernels():
  program{buildProgramFromString(FAFS_PROGRAM)},
  fill{createKernelFunctor<fillKernel>(program, "fill")},
  applyVonNeumannBC_x{createKernelFunctor<vonNeumannKernel>(program, "applyVonNeumannBC_x")},
  applyVonNeumannBC_y{createKernelFunctor<vonNeumannKernel>(program, "applyVonNeumannBC_y")},
  advanceEuler{createKernelFunctor<advanceEulerKernel>(program, "advanceEuler")}
{}
