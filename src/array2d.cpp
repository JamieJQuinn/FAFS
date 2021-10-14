#include <iostream>
#include <H5Cpp.h>

#include <array2d.hpp>
#include <constants.hpp>

Array::Array(const Constants& c_in, const std::string& name, real initialVal):
  c{c_in}
{
  setName(name);
  data = new real[size()];
  initialise(initialVal);
}

Array::Array(const Constants& c_in, real initialVal):
  c{c_in},
  name{""},
  hasName{false}
{
  data = new real[size()];
  initialise(initialVal);
}

void Array::initialise(real initialVal) {
  for (int i=0; i<size(); ++i) {
    data[i] = initialVal;
  }
}

Array::~Array() {
  delete data;
}

int Array::size() const {
  // 1 set of ghost cells at each boundary
  return (c.nx+2*c.ng)*(c.ny+2*c.ng);
}

void Array::render() const {
  for (int j=c.ny-1; j>=0; --j) {
    for (int i=0; i<c.nx; ++i) {
      std::cout << int((*this)(i,j));
    }
    std::cout << std::endl;
  }
}

void Array::applyKernel(kernelFn fn, Array& out) const {
#pragma omp parallel for collapse(2) schedule(static)
  for (int i=0; i<c.nx; ++i) {
    for (int j=0; j<c.ny; ++j) {
      out(i,j) = fn((*this), i, j);
    }
  }
}

void Array::applyKernel(kernelFnInPlaceInput fn, const Array& in) {
#pragma omp parallel for collapse(2) schedule(static)
  for (int i=0; i<c.nx; ++i) {
    for (int j=0; j<c.ny; ++j) {
      fn((*this), in, i, j);
    }
  }
}

void Array::applyKernel(kernelFnInPlace fn) {
#pragma omp parallel for collapse(2) schedule(static)
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

void Array::saveTo(H5::H5File& file) const {
  if (!hasName) {
    std::cerr << "Cannot save unnamed Array" << std::endl;
  }
  hsize_t dims[2];
  dims[0] = c.nx + 2*c.ng;
  dims[1] = c.ny + 2*c.ng;
  H5::DataSpace dataspace(2, dims);
  H5::FloatType datatype(H5::PredType::NATIVE_DOUBLE);
  datatype.setOrder(H5T_ORDER_LE);
  H5::DataSet ds = file.createDataSet(name, datatype, dataspace);
  ds.write(data, H5::PredType::NATIVE_DOUBLE);
}

void Array::setName(const std::string& name) {
  this->name = name;
  hasName = true;
}

void Array::swap(Array& arr) {
  std::swap(this->data, arr.data);
  std::swap(this->name, arr.name);
}

void Array::swapData(Array& arr) {
  std::swap(this->data, arr.data);
}
