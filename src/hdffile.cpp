#include <hdffile.hpp>

HDFFile::HDFFile(const std::string& name_in):
  name{name_in},
  file{H5::H5File(name_in, H5F_ACC_TRUNC)},
  isOpen{true}
{}

void HDFFile::close() {
  if (isOpen) {
    file.close();
    isOpen = false;
  }
}
