#include <iostream>
#include <memory>
#include "variables.hpp"
#include "precision.hpp"

int NX = 10;
int NY = 10;
int NG = 1; // Number of ghost cells
int ARR_SIZE = (NX+2*NG)*(NY+2*NG);

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

int main() {
  std::cout << "NX: " << NX << std::endl;
  std::cout << "NY: " << NY << std::endl;
  std::cout << "NG: " << NG << std::endl;

  Variables vars(ARR_SIZE);
  setInitialConditions(vars);

  real t=0;
  real total_time = 1.0;
  real dt = 0.5;

  while (t < total_time) {
    std::cout << "Still running" << std::endl;
    t += dt;
  }
  render(vars);
}
