#include <iostream>
#include <memory>

typedef double real;

int NX = 100;
int NY = 100;
int NG = 1; // Number of ghost cells
int ARR_SIZE = (NX+NG)*(NY+NG);

auto vx = std::make_unique<real[]>(ARR_SIZE);

int main() {
  std::cout << "NX: " << NX << std::endl;
  std::cout << "NY: " << NY << std::endl;
  std::cout << "NG: " << NG << std::endl;

  real t=0;
  real total_time = 1.0;
  real dt = 0.5;

  while (t < total_time) {
    std::cout << "Still running" << std::endl;
    t += dt;
  }
}
