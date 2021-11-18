#include <kernels.hpp>

Kernels::Kernels():
  program{buildProgramFromString(FAFS_PROGRAM)},
  fill{createKernelFunctor<fillKernel>(program, "fill")},
  applyVonNeumannBC_x{createKernelFunctor<vonNeumannKernel>(program, "applyVonNeumannBC_x")},
  applyVonNeumannBC_y{createKernelFunctor<vonNeumannKernel>(program, "applyVonNeumannBC_y")},
  advanceEuler{createKernelFunctor<advanceEulerKernel>(program, "advanceEuler")},
  calcDiffusionTerm{createKernelFunctor<calcDiffusionKernel>(program, "calcDiffusionTerm")},
  calcAdvectionTerm{createKernelFunctor<calcAdvectionKernel>(program, "calcAdvectionTerm")}
{}

const std::string FAFS_PROGRAM{R"CLC(
typedef float real;

int gid(int i, int j, int nx, int ny, int ng) {
  return i*(ny+2*ng) + j;
}

__kernel void fill(
  __global real *out,
  __private const real val,
  __private const int nx,
  __private const int ny,
  __private const int ng
)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int idx = gid(i, j, nx, ny, ng);
  out[idx] = val;
}

// Apply von Neumman boundary conditions to upper and lower boundaries
__kernel void applyVonNeumannBC_y(
  __global real *out,
  __private const int nx,
  __private const int ny,
  __private const int ng
)
{
  int i = get_global_id(0);

  // Set lower boundary
  int i_boundary = gid(i, 0, nx, ny, ng);
  int i_interior = gid(i, 1, nx, ny, ng);
  out[i_boundary] = out[i_interior];

  // Set upper boundary
  i_boundary = gid(i, ny+ng, nx, ny, ng);
  i_interior = gid(i, ny, nx, ny, ng);
  out[i_boundary] = out[i_interior];
}

__kernel void applyVonNeumannBC_x(
  __global real *out,
  __private const int nx,
  __private const int ny,
  __private const int ng
)
{
  int j = get_global_id(1);

  // Set lower boundary
  int i_boundary = gid(0, j, nx, ny, ng);
  int i_interior = gid(1, j, nx, ny, ng);
  out[i_boundary] = out[i_interior];

  // Set upper boundary
  i_boundary = gid(nx+ng, j, nx, ny, ng);
  i_interior = gid(nx, j, nx, ny, ng);
  out[i_boundary] = out[i_interior];
}

__kernel void advanceEuler(
  __global real *out,
  __global const real *ddt,
  __private const real dt,
  __private const int nx,
  __private const int ny,
  __private const int ng
)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int idx = gid(i, j, nx, ny, ng);
  out[idx] += ddt[idx]*dt;
}

real ddx(const real *f, const real dx, const int i, const int j, const int nx, const int ny, const int ng) {
  int ip = gid(i+1, j, nx, ny, ng);
  int im = gid(i-1, j, nx, ny, ng);
  return (f[ip]-f[im])/(2.0f*dx);
}

real ddy(const real *f, const real dy, const int i, const int j, const int nx, const int ny, const int ng) {
  int jp = gid(i, j+1, nx, ny, ng);
  int jm = gid(i, j-1, nx, ny, ng);
  return (f[jp]-f[jm])/(2.0f*dy);
}

__kernel void calcAdvectionTerm(
  __global real *out,
  __global const real *f,
  __global const real *vx,
  __global const real *vy,
  __private const real dx,
  __private const real dy,
  __private const int nx,
  __private const int ny,
  __private const int ng
)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int idx = gid(i, j, nx, ny, ng);
  out[idx] = -(vx[idx] * ddx(f, dx, i, j, nx, ny, ng) + vy[idx] * ddy(f, dy, i, j, nx, ny, ng));
}

__kernel void calcDiffusionTerm(
  __global real *out,
  __global const real *f,
  __private const real dx,
  __private const real dy,
  __private const int nx,
  __private const int ny,
  __private const int ng
)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int ij = gid(i, j, nx, ny, ng);
  int ipj = gid(i+1, j, nx, ny, ng);
  int imj = gid(i-1, j, nx, ny, ng);
  int ijp = gid(i, j+1, nx, ny, ng);
  int ijm = gid(i, j-1, nx, ny, ng);
  out[ij] = (f[ijp] + f[ijm] + f[ipj] + f[imj] - 4*f[ij])/(dx*dy);
}
)CLC"};
