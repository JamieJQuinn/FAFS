#include <iostream>

#include <array2d.hpp>
#include <constants.hpp>

Array::Array(const Constants& c_in, real initial_val):
  c{c_in}
{
  data = std::make_unique<real[]>(size());
  for (int i=0; i<size(); ++i) {
    data[i] = initial_val;
  }
}

int Array::size() const {
  return (c.nx+2*c.ng)*(c.ny+2*c.ng);
}

void Array::render() const {
  for (int j=0; j<c.ny; ++j) {
    for (int i=0; i<c.nx; ++i) {
      std::cout << int((*this)(i,j)) << " ";
    }
    std::cout << std::endl;
  }
}

void Array::applyKernel(kernelFn fn, Array& out) const {
  for (int i=0; i<c.nx; ++i) {
    for (int j=0; j<c.ny; ++j) {
      out(i,j) = fn((*this), i, j);
    }
  }
}

void Array::applyKernel(kernelFnInPlace fn) {
  for (int i=0; i<c.nx; ++i) {
    for (int j=0; j<c.ny; ++j) {
      fn((*this), i, j);
    }
  }
}

const int Array::idx(const int i, const int j) const {
  return (i+c.ng)*(c.ny+2*c.ng) + (j+c.ng);
}

const real Array::operator()(const int i, const int j) const {
  return data[idx(i,j)];
}

real& Array::operator()(const int i, const int j) {
  return data[idx(i,j)];
}

Array& Array::operator=(const Array& arr) {
  for (int i=0; i<c.nx; ++i) {
    for (int j=0; j<c.ny; ++j) {
      (*this)(i,j) = arr(i,j);
    }
  }
  return *this;
}

void Array::operator+=(const Array& arr) {
  for (int i=0; i<c.nx; ++i) {
    for (int j=0; j<c.ny; ++j) {
      (*this)(i,j) += arr(i,j);
    }
  }
}
