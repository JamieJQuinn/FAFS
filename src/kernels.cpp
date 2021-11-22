#include <kernels.hpp>

Kernels::Kernels():
  program{buildProgramFromString(FAFS_PROGRAM)},
  fill{createKernelFunctor<fillKernel>(program, "fill")},
  applyVonNeumannBC_x{createKernelFunctor<vonNeumannKernel>(program, "applyVonNeumannBC_x")},
  applyVonNeumannBC_y{createKernelFunctor<vonNeumannKernel>(program, "applyVonNeumannBC_y")},
  advanceEuler{createKernelFunctor<advanceEulerKernel>(program, "advanceEuler")},
  calcDiffusionTerm{createKernelFunctor<calcDiffusionKernel>(program, "calcDiffusionTerm")},
  calcAdvectionTerm{createKernelFunctor<calcAdvectionKernel>(program, "calcAdvectionTerm")},
  applyJacobiStep{createKernelFunctor<applyJacobiKernel>(program, "applyJacobiStep")},
  calcDivergence{createKernelFunctor<calcDivergence_k>(program, "calcDivergence")},
  applyProjectionX{createKernelFunctor<applyProjection_k>(program, "applyProjectionX")},
  applyProjectionY{createKernelFunctor<applyProjection_k>(program, "applyProjectionY")}
{}

const std::string FAFS_PROGRAM{R"CLC(
typedef float real;

int index(int i, int j, int nx, int ny, int ng) {
  return (i+ng)*(ny+2*ng) + (j+ng);
}

int gid(int i, int ng) {
  return get_global_id(i) - ng;
}

__kernel void fill(
  __global real *out,
  __private const real val,
  __private const int nx,
  __private const int ny,
  __private const int ng
)
{
  int i = gid(0, ng);
  int j = gid(1, ng);
  int idx = index(i, j, nx, ny, ng);
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
  int i = gid(0, ng);

  // Set lower boundary
  int i_boundary = index(i, -1, nx, ny, ng);
  int i_interior = index(i, 0, nx, ny, ng);
  out[i_boundary] = out[i_interior];

  // Set upper boundary
  i_boundary = index(i, ny, nx, ny, ng);
  i_interior = index(i, ny-1, nx, ny, ng);
  out[i_boundary] = out[i_interior];
}

__kernel void applyVonNeumannBC_x(
  __global real *out,
  __private const int nx,
  __private const int ny,
  __private const int ng
)
{
  int j = gid(1, ng);

  // Set lower boundary
  int i_boundary = index(-1, j, nx, ny, ng);
  int i_interior = index(0, j, nx, ny, ng);
  out[i_boundary] = out[i_interior];

  // Set upper boundary
  i_boundary = index(nx, j, nx, ny, ng);
  i_interior = index(nx-1, j, nx, ny, ng);
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
  int i = gid(0, ng);
  int j = gid(1, ng);
  int idx = index(i, j, nx, ny, ng);
  out[idx] += ddt[idx]*dt;
}

real ddx(const real *f, const real dx, const int i, const int j, const int nx, const int ny, const int ng) {
  int ip = index(i+1, j, nx, ny, ng);
  int im = index(i-1, j, nx, ny, ng);
  return (f[ip]-f[im])/(2.0f*dx);
}

real ddy(const real *f, const real dy, const int i, const int j, const int nx, const int ny, const int ng) {
  int jp = index(i, j+1, nx, ny, ng);
  int jm = index(i, j-1, nx, ny, ng);
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
  int i = gid(0, ng);
  int j = gid(1, ng);
  int idx = index(i, j, nx, ny, ng);
  out[idx] = -(vx[idx] * ddx(f, dx, i, j, nx, ny, ng) + vy[idx] * ddy(f, dy, i, j, nx, ny, ng));
}

__kernel void calcDiffusionTerm(
  __global real *out,
  __global const real *f,
  __private const real dx,
  __private const real dy,
  __private const real Re,
  __private const int nx,
  __private const int ny,
  __private const int ng
)
{
  int i = gid(0, ng);
  int j = gid(1, ng);
  int ij = index(i, j, nx, ny, ng);
  int ipj = index(i+1, j, nx, ny, ng);
  int imj = index(i-1, j, nx, ny, ng);
  int ijp = index(i, j+1, nx, ny, ng);
  int ijm = index(i, j-1, nx, ny, ng);
  out[ij] = (f[ijp] + f[ijm] + f[ipj] + f[imj] - 4*f[ij])/(Re*dx*dy);
}

__kernel void applyJacobiStep(
  __global real *out,
  __global const real *in,
  __private const real alpha,
  __private const real beta,
  __global const real *b,
  __private const int nx,
  __private const int ny,
  __private const int ng
)
{
  int i = gid(0, ng);
  int j = gid(1, ng);

  int ij = index(i, j, nx, ny, ng);
  int ipj = index(i+1, j, nx, ny, ng);
  int imj = index(i-1, j, nx, ny, ng);
  int ijp = index(i, j+1, nx, ny, ng);
  int ijm = index(i, j-1, nx, ny, ng);

  out[ij] = (alpha*b[ij] + in[ijp] + in[ijm] + in[ipj] + in[imj])/beta;
}

__kernel void calcDivergence(
  __global real *out,
  __global const real *fx,
  __global const real *fy,
  __private const real dx,
  __private const real dy,
  __private const int nx,
  __private const int ny,
  __private const int ng
)
{
  int i = gid(0, ng);
  int j = gid(1, ng);

  int vij = index(i, j, nx-1, ny-1, ng);
  int vimj = index(i-1, j, nx-1, ny-1, ng);
  int vijm = index(i, j-1, nx-1, ny-1, ng);

  int ij = index(i, j, nx, ny, ng);

  out[ij] = (fx[vij] - fx[vimj])/dx + (fy[vij] - fy[vijm])/dy;
}

__kernel void applyProjectionX(
  __global real *out,
  __global const real *f,
  __private const real dx,
  __private const int nx,
  __private const int ny,
  __private const int ng
)
{
  int i = gid(0, ng);
  int j = gid(1, ng);

  int ij = index(i, j, nx, ny, ng);

  int cij =  index(i  , j, nx+1, ny+1, ng);
  int cipj = index(i+1, j, nx+1, ny+1, ng);

  out[ij] = out[ij] - (f[cipj] - f[cij])/dx;
}

__kernel void applyProjectionY(
  __global real *out,
  __global const real *f,
  __private const real dy,
  __private const int nx,
  __private const int ny,
  __private const int ng
)
{
  int i = gid(0, ng);
  int j = gid(1, ng);

  int ij = index(i, j, nx, ny, ng);

  int cij =  index(i, j  , nx+1, ny+1, ng);
  int cijp = index(i, j+1, nx+1, ny+1, ng);

  out[ij] = out[ij] - (f[cijp] - f[cij])/dy;
}
)CLC"};

Kernels g_kernels; // wuh oh, is that a global variable? it is, deal with it
