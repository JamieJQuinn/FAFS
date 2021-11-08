#include <iostream>
#include <H5Cpp.h>

#include <array2d.hpp>
#include <constants.hpp>

Array::Array(const int nx_in, const int ny_in, const int ng_in, const std::string& name_in, real initialVal):
  nx{nx_in},
  ny{ny_in},
  ng{ng_in},
  hasName{name_in != ""},
  data{nullptr}
{
  setName(name_in);
  data = new real[size()];
  initialise(initialVal);
}

void Array::initialise(real initialVal) {
  for (int i=0; i<size(); ++i) {
    data[i] = initialVal;
  }
}

Array::~Array() {
  if(data != nullptr) {
    delete data;
  }
}

int Array::size() const {
  // 1 set of ghost cells at each boundary
  return (nx+2*ng)*(ny+2*ng);
}

void Array::render() const {
  for (int j=ny-1; j>=0; --j) {
    for (int i=0; i<nx; ++i) {
      std::cout << int((*this)(i,j));
    }
    std::cout << std::endl;
  }
}

void Array::applyKernel(kernelFn fn, Array& out) const {
#pragma omp parallel for collapse(2) schedule(static)
  for (int i=0; i<nx; ++i) {
    for (int j=0; j<ny; ++j) {
      out(i,j) = fn((*this), i, j);
    }
  }
}

real Array::sum() const {
  int result = 0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+:result)
  for (int i=0; i<nx; ++i) {
    for (int j=0; j<ny; ++j) {
      result += (*this)(i,j);
    }
  }
}

void Array::applyKernel(kernelFnInPlaceInput fn, const Array& in) {
#pragma omp parallel for collapse(2) schedule(static)
  for (int i=0; i<nx; ++i) {
    for (int j=0; j<ny; ++j) {
      fn((*this), in, i, j);
    }
  }
}

void Array::applyKernel(kernelFnInPlace fn) {
#pragma omp parallel for collapse(2) schedule(static)
  for (int i=0; i<nx; ++i) {
    for (int j=0; j<ny; ++j) {
      fn((*this), i, j);
    }
  }
}

const int Array::idx(const int i, const int j) const {
  return (i+ng)*(ny+2*ng) + (j+ng);
}

const real Array::operator()(const int i, const int j) const {
  return data[idx(i,j)];
}

real& Array::operator()(const int i, const int j) {
  return data[idx(i,j)];
}

Array& Array::operator=(const Array& arr) {
  for (int i=0; i<nx; ++i) {
    for (int j=0; j<ny; ++j) {
      (*this)(i,j) = arr(i,j);
    }
  }
  return *this;
}

void Array::operator+=(const Array& arr) {
  for (int i=0; i<nx; ++i) {
    for (int j=0; j<ny; ++j) {
      (*this)(i,j) += arr(i,j);
    }
  }
}

void Array::saveTo(H5::H5File& file) const {
  if (!hasName) {
    std::cerr << "Cannot save unnamed Array" << std::endl;
  }
  hsize_t dims[2];
  dims[0] = nx + 2*ng;
  dims[1] = ny + 2*ng;
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
