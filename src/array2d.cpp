#include <iostream>
#include <H5Cpp.h>

#include <array2d.hpp>
#include <constants.hpp>

Array::Array(const int nx_in, const int ny_in, const int ng_in, const std::string& name_in, real initialVal):
  nx{nx_in},
  ny{ny_in},
  ng{ng_in},
  data(size()),
  h5ArrayType(H5::PredType::NATIVE_FLOAT)
{
  setName(name_in);
  fill(initialVal);
}

void Array::fill(real val) {
  for (int i=0; i<size(); ++i) {
    data[i] = val;
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
    throw std::runtime_error("Cannot save unnamed Array");
  }
  hsize_t dims[2];
  dims[0] = nx + 2*ng;
  dims[1] = ny + 2*ng;
  H5::DataSpace dataspace(2, dims);
  H5::FloatType datatype(h5ArrayType);
  datatype.setOrder(H5T_ORDER_LE);
  H5::DataSet ds = file.createDataSet(name.c_str(), datatype, dataspace);
  ds.write(data.data(), h5ArrayType);
}

void Array::load(H5::H5File& file) {
  if (!hasName) {
    throw std::runtime_error("Cannot load unnamed Array");
  }
  H5::DataSet ds = file.openDataSet(name.c_str());
  H5::DataSpace filespace = ds.getSpace();
  int ndims = filespace.getSimpleExtentNdims();
  hsize_t dims[2];
  filespace.getSimpleExtentDims(dims);

  H5::DataSpace memspace (ndims, dims);
  ds.read(data.data(), h5ArrayType, memspace, filespace);
}

void Array::setName(const std::string& name) {
  this->name = name;
  hasName = name != "";
}

void Array::swap(Array& arr) {
  std::swap(data, arr.data);
  std::swap(name, arr.name);
}

void Array::swapData(Array& arr) {
  std::swap(data, arr.data);
}

void Array::print() const {
  for(int i=0; i<nx; ++i) {
    for(int j=0; j<nx; ++j) {
      std::cout << (*this)(i,j) << ", ";
    }
    std::cout << std::endl;
  }
}
