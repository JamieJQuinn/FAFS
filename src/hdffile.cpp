#include <stdexcept>
#include <hdffile.hpp>

HDFFile::HDFFile(const std::string& name_in):
  name{name_in},
  isOpen{false}
{
  open(name_in);
}

void HDFFile::open(const std::string& name) {
  if(isOpen) {
    throw std::runtime_error("Cannot open " + name + ": File not closed!");
  }
  file = H5::H5File(name.c_str(), H5F_ACC_TRUNC);
  isOpen = true;
}

void HDFFile::close() {
  if (isOpen) {
    file.close();
    isOpen = false;
  }
}
