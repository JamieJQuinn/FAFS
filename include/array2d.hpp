#pragma once

#include <memory>
#include <precision.hpp>

typedef std::unique_ptr<real[]> array;

array makeArray(const int nx, const int ny, const int ng);
int calcTotalArraySize(const int nx, const int ny, const int ng);
