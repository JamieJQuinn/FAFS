#pragma once

#include <string>
#include <CL/opencl.hpp>

cl::Program buildProgram(const std::string& filename);
int setDefaultPlatform(const std::string& targetName);
