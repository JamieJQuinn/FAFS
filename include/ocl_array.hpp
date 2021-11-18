#pragma once

#include <ocl_utility.hpp>
#include <array2d.hpp>
#include <precision.hpp>

class openCLArray: public Array {
  public:
    openCLArray(const int nx, const int ny, const int ng = 0, const std::string& name = "", real initialVal = 0.0f, bool initDevice = true);
    void initOnDevice(bool readOnly = false);
    std::vector<real>::iterator begin();
    std::vector<real>::iterator end();
    const cl::Buffer& getDeviceData() const;
    cl::Buffer& getDeviceData();
    //const cl::EnqueueArgs& makeRange() const;

    void toDevice();
    void toHost();

    cl::EnqueueArgs range;
    cl::EnqueueArgs entireRange;
    cl::EnqueueArgs lowerBRange;
    cl::EnqueueArgs upperBRange;
    cl::EnqueueArgs leftBRange;
    cl::EnqueueArgs rightBRange;

  protected:
    cl::Buffer d_data; // data on device
    bool isDeviceDirty;
};
