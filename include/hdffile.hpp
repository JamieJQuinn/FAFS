#pragma once

#include <string>
#include <H5Cpp.h>

class HDFFile {
  public:
    HDFFile(const std::string& name, const bool readOnly = true);
    void open(const std::string& name, const bool readOnly = true);
    void close();
    H5::H5File file;
  private:
    const std::string name;
    bool isOpen;
};
