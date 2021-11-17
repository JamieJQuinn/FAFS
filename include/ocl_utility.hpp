#pragma once

#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <string>
#include <CL/opencl.hpp>

cl::Program buildProgram(const std::string& filename);
int setDefaultPlatform(const std::string& targetName);
