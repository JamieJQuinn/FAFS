#pragma once

#include <ocl_utility.hpp>
#include <array2d.hpp>
#include <precision.hpp>
#include <kernels.hpp>

class openCLArray: public Array {
  public:
    openCLArray(const int nx, const int ny, const int ng = 0, const std::string& name = "", real initialVal = 0.0f, bool initDevice = true);
    void initOnDevice(bool readOnly = false);
    std::vector<real>::iterator begin();
    std::vector<real>::iterator end();
    const cl::Buffer& getDeviceData() const;
    cl::Buffer& getDeviceData();
    const cl::EnqueueArgs makeRange(int x0, int x1, int y0, int y1) const;
    const cl::EnqueueArgs makeColumnRange(int col, bool includeGhost=false) const;
    const cl::EnqueueArgs makeRowRange(int row, bool includeGhost=false) const;
    void swapData(openCLArray& arr);
    void fill(real val, bool includeGhost = false);

    void toDevice();
    void toHost();

    cl::EnqueueArgs interior;
    cl::EnqueueArgs entire;
    cl::EnqueueArgs lowerBound;
    cl::EnqueueArgs upperBound;
    cl::EnqueueArgs leftBound;
    cl::EnqueueArgs rightBound;

  protected:
    cl::Buffer d_data; // data on device
    bool isDeviceDirty;
    cl::Program program;
    fillKernel fill_k;
};
