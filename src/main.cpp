#include <iostream>
#include <memory>

#include <array2d.hpp>
#include <variables.hpp>
#include <precision.hpp>

void setInitialConditions(Variables& vars, const Constants& c) {
  for (int i=0; i<c.nx/2; ++i) {
    for (int j=0; j<c.ny; ++j) {
      vars.vx(i,j) = 9.0;
    }
  }
}

void render(const Variables& vars) {
  vars.vx.render();
}

real diffusionKernel(const Array& f, const int i, const int j, const Constants& c) {
  return (f(i,j+1) + f(i,j-1) + f(i+1,j) + f(i-1,j) - 4*f(i,j))/(c.dx*c.dy);
}

int main() {
  const Constants c;

  Variables vars(c);

  setInitialConditions(vars, c);

  Array out(c);
  out.render();
  vars.vx.applyKernel(out, diffusionKernel);

  vars.vx.render();
  out.render();

  real t=0;
  real total_time = 1.0;
  real dt = 0.5;

  while (t < total_time) {
    std::cout << "Still running" << std::endl;
    t += dt;
  }
}
