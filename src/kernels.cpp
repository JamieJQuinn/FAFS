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
  applyProjectionY{createKernelFunctor<applyProjection_k>(program, "applyProjectionY")},
  advect{createKernelFunctor<advect_k>(program, "advect")}
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
  __private const real gamma,
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

  out[ij] = alpha*(b[ij] - (in[ipj] + in[imj])/beta - (in[ijp] + in[ijm])/gamma);
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
  // cell-centred global indices
  int i = gid(0, ng);
  int j = gid(1, ng);

  // node-centred array indices
  int vij   = index(i,   j,   nx-1, ny-1, ng);
  int vijm  = index(i,   j-1, nx-1, ny-1, ng);
  int vimj  = index(i-1, j,   nx-1, ny-1, ng);
  int vimjm = index(i-1, j-1, nx-1, ny-1, ng);

  // cell-centred array index
  int ij = index(i, j, nx, ny, ng);

  real dvxj  = (fx[vij ] - fx[vimj ])/dx; // ddx at upper boundary
  real dvxjm = (fx[vijm] - fx[vimjm])/dx; // ddx at lower boundary
  real dvxdx = 0.5*(dvxj + dvxjm); // ddx at cell centre

  real dvyi  = (fy[vij ] - fy[vijm ])/dy; // ddy at right boundary
  real dvyim = (fy[vimj] - fy[vimjm])/dy; // ddy at left boundary
  real dvydy = 0.5*(dvyi + dvyim); // ddy at cell centre

  out[ij] = dvydy + dvxdx;
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

  // Cell-centred array indices
  int cij   = index(i,   j,   nx+1, ny+1, ng);
  int cijp  = index(i,   j+1, nx+1, ny+1, ng);
  int cipj  = index(i+1, j,   nx+1, ny+1, ng);
  int cipjp = index(i+1, j+1, nx+1, ny+1, ng);

  real dfdxjp = (f[cipjp] - f[cijp])/dx; // ddx at upper boundary
  real dfdxj  = (f[cipj ] - f[cij ])/dx; // ddx at lower boundary
  real dfdx = 0.5*(dfdxj + dfdxjp); // ddx at node

  out[ij] = out[ij] - dfdx;
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

  // Cell-centred array indices
  int cij   = index(i,   j,   nx+1, ny+1, ng);
  int cijp  = index(i,   j+1, nx+1, ny+1, ng);
  int cipj  = index(i+1, j,   nx+1, ny+1, ng);
  int cipjp = index(i+1, j+1, nx+1, ny+1, ng);

  real dfdyip = (f[cipjp] - f[cipj])/dy; // ddy at right boundary
  real dfdyi  = (f[cijp ] - f[cij ])/dy; // ddy at left boundary
  real dfdy = 0.5*(dfdyi + dfdyip); // ddy at node

  out[ij] = out[ij] - dfdy;
}

__kernel void advect(
  __global real *out,
  __global const real *f,
  __global const real *vx,
  __global const real *vy,
  __private const real dx,
  __private const real dy,
  __private const real dt,
  __private const int nx,
  __private const int ny,
  __private const int ng
) {
  int i = gid(0, ng);
  int j = gid(1, ng);

  int ij = index(i, j, nx, ny, ng);

  real x = (real)i - dt*vx[ij]/dx;
  real y = (real)j - dt*vy[ij]/dy;
  // clamp to int indices and ensure inside domain
  int rigIdx = nx-1+ng;
  int lefIdx = -ng;
  int topIdx = ny-1+ng;
  int botIdx = -ng;
  int x2 = clamp((int)floor(x+1),lefIdx, rigIdx);
  int x1 = clamp((int)floor(x)  ,lefIdx, rigIdx);
  int y2 = clamp((int)floor(y+1),botIdx, topIdx);
  int y1 = clamp((int)floor(y)  ,botIdx, topIdx);
  x = clamp(x, (real)lefIdx, (real)rigIdx);
  y = clamp(y, (real)botIdx, (real)topIdx);
  // bilinearly interpolate
  real fy1, fy2;
  int x1y1 = index(x1, y1, nx, ny, ng);
  int x1y2 = index(x1, y2, nx, ny, ng);
  if(x1!=x2) {
    real x1Weight = (x2-x)/(x2-x1);
    real x2Weight = (x-x1)/(x2-x1);

    int x2y1 = index(x2, y1, nx, ny, ng);
    int x2y2 = index(x2, y2, nx, ny, ng);

    fy1 = x1Weight*f[x1y1] + x2Weight*f[x2y1];
    fy2 = x1Weight*f[x1y2] + x2Weight*f[x2y2];
  } else {
    fy1 = f[x1y1];
    fy2 = f[x1y2];
  }
  real fAv;
  if(y1!=y2) {
    real y1Weight = (y2-y)/(y2-y1);
    real y2Weight = (y-y1)/(y2-y1);
    fAv = y1Weight*fy1 + y2Weight*fy2;
  } else {
    fAv = fy1;
  }
  out[ij] = fAv;
}
)CLC"};

Kernels g_kernels; // wuh oh, is that a global variable? it is, deal with it
