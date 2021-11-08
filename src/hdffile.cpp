#include <hdffile.hpp>

HDFFile::HDFFile(const std::string& name_in):
  file{H5::H5File(name_in.c_str(), H5F_ACC_TRUNC)},
  name{name_in},
  isOpen{true}
{}

void HDFFile::close() {
  if (isOpen) {
    file.close();
    isOpen = false;
  }
}
