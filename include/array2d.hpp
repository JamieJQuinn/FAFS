#pragma once

#include <memory>
#include <functional>

#include <precision.hpp>
#include <constants.hpp>

class Array;

typedef std::function<real(const Array&, const int, const int)> kernelFn;
typedef std::function<void(Array&, const int, const int)> kernelFnInPlace;

class Array {
  public:
    std::unique_ptr<real[]> data;
    const Constants& c;

    Array(const Constants& c_in, real initial_val = 0.0f);
    const int idx(const int i, const int j) const;
    int size() const;
    void render() const;
    void applyKernel(kernelFn fn, Array& out) const;
    void applyKernel(kernelFnInPlace fn);
    const real operator()(const int i, const int j) const;
    real& operator()(const int i, const int j);
    Array& operator=(const Array& arr);
    void operator+=(const Array& arr);
};
