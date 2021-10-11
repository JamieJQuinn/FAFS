#include "variables.hpp"

Variables::Variables(const int arr_size):
  vx{std::make_unique<real[]>(arr_size)}
{}
