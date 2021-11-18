#include <ocl_array.hpp>

openCLArray::openCLArray(const int nx, const int ny, const int ng, const std::string& name, real initialVal, bool initDevice):
  Array(nx, ny, ng, name, initialVal),
  isDeviceDirty{true},
  range(cl::NDRange(ng, ng), cl::NDRange(nx, ny), cl::NDRange(nx, ny)),
  entireRange(cl::NDRange(0, 0), cl::NDRange(nx+2*ng, ny+2*ng), cl::NDRange(nx+2*ng, ny+2*ng)),
  lowerBRange(cl::NDRange(0,0), cl::NDRange(nx+2*ng, 1), cl::NDRange(nx+2*ng, 1)),
  upperBRange(cl::NDRange(0,ny+ng), cl::NDRange(nx+2*ng, 1), cl::NDRange(nx+2*ng, 1)),
  leftBRange(cl::NDRange(0,0), cl::NDRange(1, ny+2*ng), cl::NDRange(1, ny+2*ng)),
  rightBRange(cl::NDRange(nx+ng,0), cl::NDRange(1, ny+2*ng), cl::NDRange(1, ny+2*ng))
{
  if(initDevice) {
    initOnDevice();
  }
}

void openCLArray::initOnDevice(bool readOnly) {
  d_data = cl::Buffer(begin(), end(), readOnly);
}

std::vector<real>::iterator openCLArray::begin() {
  return data.begin();
}

std::vector<real>::iterator openCLArray::end() {
  isDeviceDirty = true;
  return data.end();
}

const cl::Buffer& openCLArray::getDeviceData() const {
  return d_data;
}

cl::Buffer& openCLArray::getDeviceData() {
  return d_data;
}

void openCLArray::toDevice() {
  cl::copy(begin(), end(), d_data);
}

void openCLArray::toHost() {
  cl::copy(d_data, begin(), end());
  isDeviceDirty = false;
}
