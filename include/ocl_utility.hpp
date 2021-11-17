#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS 
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <string>
#include <CL/opencl.hpp>

cl::Program buildProgramFromFile(const std::string& filename);
cl::Program buildProgramFromString(const std::string& source);
int setDefaultPlatform(const std::string& targetName);
