#pragma once

#include <string>
#include <memory>
#include <functional>
#include <H5Cpp.h>

#include <ocl_utility.hpp>
#include <precision.hpp>
#include <constants.hpp>

class Array;

typedef std::function<real(const Array&, const int, const int)> kernelFn;
typedef std::function<void(Array&, const Array&, const int, const int)> kernelFnInPlaceInput;
typedef std::function<void(Array&, const int, const int)> kernelFnInPlace;

class Array {
  public:
    Array(const int nx, const int ny, const int ng = 0, const std::string& name = "", real initialVal = 0.0f);
    void initOnDevice(bool readOnly = false);
    void initialise(real initialVal);
    int idx(const int i, const int j) const;
    int size() const;
    void applyKernel(kernelFn fn, Array& out) const;
    void applyKernel(kernelFnInPlaceInput fn, const Array& in);
    void applyKernel(kernelFnInPlace fn);
    real sum() const;
    real operator()(const int i, const int j) const;
    real& operator()(const int i, const int j);
    Array& operator=(const Array& arr);
    void operator+=(const Array& arr);
    void saveTo(H5::H5File& file) const;
    void setName(const std::string& name);
    void swap(Array& arr);
    void swapData(Array& arr);
    std::vector<real>::iterator begin();
    std::vector<real>::iterator end();
    const cl::Buffer& getDeviceData() const;
    cl::Buffer& getDeviceData();

    void toDevice();
    void toHost();

    const int nx, ny, ng;
    cl::EnqueueArgs range;
  private:
    std::vector<real> data;
    cl::Buffer d_data; // data on device
    std::string name;
    bool hasName;
    bool isDeviceDirty;
};
