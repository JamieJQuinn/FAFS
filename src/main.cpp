#include <iostream>
#include <memory>

#include <array2d.hpp>
#include <variables.hpp>
#include <precision.hpp>

int NX = 10;
int DX = 1.0/(NX+1);
int NY = 10;
int DY = 1.0/(NY+1);
int NG = 1; // Number of ghost cells

inline int idx(const int i, const int j) {
  return (i+NG)*(NY+2*NG) + (j+NG);
}

void setInitialConditions(Variables& vars) {
  for (int i=0; i<NX/2; ++i) {
    for (int j=0; j<NY; ++j) {
      vars.vx[idx(i,j)] = 1.0;
    }
  }
}

void render(const Variables& vars) {
  for (int j=0; j<NY; ++j) {
    for (int i=0; i<NX; ++i) {
      std::cout << int(vars.vx[idx(i,j)]);
    }
    std::cout << std::endl;
  }
}

real diffusionKernel(const array& f, const int i, const int j) {
  return (f[idx(i,j+1)] + f[idx(i,j-1)] + f[idx(i+1,j)] + f[idx(i-1,j)] - 4*f[idx(i,j)])/(DX*DY);
}

void applyKernel(const array& f, array& temp, kernelFn fn) {
  for (int i=0; i<NX; ++i) {
    for (int j=0; j<NY; ++j) {
      temp[idx(i,j)] = fn(f, i, j);
    }
  }
}

int main() {
  std::cout << "NX: " << NX << std::endl;
  std::cout << "NY: " << NY << std::endl;
  std::cout << "NG: " << NG << std::endl;

  Variables vars(NX, NY, NG);
  array temp = makeArray(NX, NY, NG);
  setInitialConditions(vars);

  applyKernel(vars.vx, temp, diffusionKernel);

  real t=0;
  real total_time = 1.0;
  real dt = 0.5;

  while (t < total_time) {
    std::cout << "Still running" << std::endl;
    t += dt;
  }
  render(vars);
}
