#include <ocl_array.hpp>
#include <kernels.hpp>

openCLArray::openCLArray(const int nx, const int ny, const int ng, const std::string& name, real initialVal, bool initDevice):
  Array(nx, ny, ng, name, initialVal),
  isDeviceDirty{true},
  interior(makeRange(0, 0, nx, ny)),
  entire(makeRange(-1, -1, nx+1, ny+1)),
  lowerBound(makeRowRange(-1, true)),
  upperBound(makeRowRange(ny, true)),
  leftBound(makeColumnRange(-1, true)),
  rightBound(makeColumnRange(nx, true))
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

const cl::EnqueueArgs openCLArray::makeRange(int x0, int y0, int x1, int y1) const {
  int xGroup = x1-x0;
  int yGroup = y1-y0;
  return cl::EnqueueArgs(cl::NDRange(x0+ng, y0+ng), cl::NDRange(x1-x0, y1-y0), cl::NDRange(xGroup, yGroup));
}

const cl::EnqueueArgs openCLArray::makeColumnRange(int col, bool includeGhost) const {
  int x0=col, y0=0, x1=col+1, y1=ny;

  if(includeGhost) {
    y0 -= ng;
    y1 += ng;
  }

  return makeRange(x0,y0,x1,y1);
}

const cl::EnqueueArgs openCLArray::makeRowRange(int row, bool includeGhost) const {
  int x0=0, y0=row, x1=nx, y1=row+1;

  if(includeGhost) {
    x0 -= ng;
    x1 += ng;
  }

  return makeRange(x0,y0,x1,y1);
}

void openCLArray::swapData(openCLArray& arr) {
  Array::swap(arr);
  std::swap(d_data, arr.d_data);
}

void openCLArray::fill(real val, bool includeGhost) {
  auto range = includeGhost ? entire : interior;
  g_kernels.fill(range, getDeviceData(), val, nx, ny, ng);
}

void openCLArray::setUpperBoundary(real val) {
  g_kernels.fill(upperBound, getDeviceData(), val, nx, ny, ng);
}

void openCLArray::setLowerBoundary(real val) {
  g_kernels.fill(lowerBound, getDeviceData(), val, nx, ny, ng);
}

void openCLArray::setLeftBoundary(real val) {
  g_kernels.fill(leftBound, getDeviceData(), val, nx, ny, ng);
}

void openCLArray::setRightBoundary(real val) {
  g_kernels.fill(rightBound, getDeviceData(), val, nx, ny, ng);
}
