#pragma once

#include <memory>
#include <functional>

typedef double real;
typedef std::unique_ptr<real[]> array;
typedef std::function<real(const array&, const int, const int)> kernelFn;
