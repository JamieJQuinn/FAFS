#pragma once

#include <string>
#include <memory>
#include <functional>
#include <H5Cpp.h>

#include <precision.hpp>
#include <constants.hpp>

class Array {
  public:
    Array(const int nx, const int ny, const int ng = 0, const std::string& name = "", real initialVal = 0.0f);
    void fill(real val);
    int idx(const int i, const int j) const;
    int size() const;
    real sum() const;
    real operator()(const int i, const int j) const;
    real& operator()(const int i, const int j);
    Array& operator=(const Array& arr);
    void operator+=(const Array& arr);
    void saveTo(H5::H5File& file) const;
    void load(H5::H5File& file);
    void setName(const std::string& name);
    void swap(Array& arr);
    void swapData(Array& arr);
    void print() const;

    const int nx, ny, ng;
  protected:
    std::vector<real> data;
    std::string name;
    bool hasName;
    H5::PredType h5ArrayType;
};
