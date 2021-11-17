#pragma once

#include <precision.hpp>

typedef cl::KernelFunctor<cl::Buffer, real, int, int, int> fillKernel;
typedef cl::KernelFunctor<cl::Buffer, int, int, int> vonNeumannKernel;

std::string FAFS_PROGRAM{R"CLC(
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
)CLC"};
