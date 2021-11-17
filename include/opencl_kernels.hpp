#pragma once

std::string FAFS_PROGRAM{R"CLC(
typedef float real;

int gid(int i, int j, int nx, int ny, int ng) {
  return i*(ny+2*ng) + j;
}

__kernel void add_one(
  __global real *out,
  __private const int nx,
  __private const int ny,
  __private const int ng
)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int idx = gid(i, j, nx, ny, ng);
  out[idx] = i;
}
)CLC"};
