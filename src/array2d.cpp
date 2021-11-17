#include <iostream>
#include <H5Cpp.h>

#include <array2d.hpp>
#include <constants.hpp>

Array::Array(const int nx_in, const int ny_in, const int ng_in, const std::string& name_in, real initialVal):
  nx{nx_in},
  ny{ny_in},
  ng{ng_in},
  data(size()),
  hasName{name_in != ""},
  isDeviceDirty{true},
  range(cl::NDRange(ng, ng), cl::NDRange(nx, ny), cl::NDRange(nx, ny))
{
  setName(name_in);
  initialise(initialVal);
}

void Array::initOnDevice(bool readOnly) {
  d_data = cl::Buffer(begin(), end(), readOnly);
}

void Array::initialise(real initialVal) {
  for (int i=0; i<size(); ++i) {
    data[i] = initialVal;
  }
}

int Array::size() const {
  // 1 set of ghost cells at each boundary
  return (nx+2*ng)*(ny+2*ng);
}

real Array::sum() const {
  real result = 0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+:result)
  for (int i=0; i<nx; ++i) {
    for (int j=0; j<ny; ++j) {
      result += (*this)(i,j);
    }
  }
  return result;
}

int Array::idx(const int i, const int j) const {
  return (i+ng)*(ny+2*ng) + (j+ng);
}

real Array::operator()(const int i, const int j) const {
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
  H5::DataSet ds = file.createDataSet(name.c_str(), datatype, dataspace);
  ds.write(data.data(), H5::PredType::NATIVE_DOUBLE);
}

void Array::setName(const std::string& name) {
  this->name = name;
  hasName = true;
}

void Array::swap(Array& arr) {
  std::swap(data, arr.data);
  std::swap(name, arr.name);
}

void Array::swapData(Array& arr) {
  std::swap(data, arr.data);
}

std::vector<real>::iterator Array::begin() {
  return data.begin();
}

std::vector<real>::iterator Array::end() {
  isDeviceDirty = true;
  return data.end();
}

const cl::Buffer& Array::getDeviceData() const {
  return d_data;
}

cl::Buffer& Array::getDeviceData() {
  return d_data;
}

void Array::toDevice() {
  cl::copy(begin(), end(), d_data);
}

void Array::toHost() {
  cl::copy(d_data, begin(), end());
  isDeviceDirty = false;
}
