#include <iostream>
#include <fstream>

#include <ocl_utility.hpp>

int setDefaultPlatform(const std::string& targetName) {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  cl::Platform plat;

  // Print out names of all available platforms
  std::cout << "Available platforms:" << std::endl;
  for (auto &p : platforms) {
    std::string platname = p.getInfo<CL_PLATFORM_NAME>();
    std::cout << platname << std::endl;
  }

  // Find first platform matching name
  for (auto &p : platforms) {
    std::string platname = p.getInfo<CL_PLATFORM_NAME>();
    if (platname.find(targetName) != std::string::npos) {
      plat = p;
      break;
    }
  }

  if (plat() == 0)  {
    std::cout << "No platform found with name " << targetName << ".\n";
    return -1;
  }

  cl::Platform newP = cl::Platform::setDefault(plat);
  if (newP != plat) {
    std::cout << "Error setting default platform.";
    return -1;
  }

  std::string platname = newP.getInfo<CL_PLATFORM_NAME>();
  std::cout << "Running on " << platname << std::endl;

  return 0;
}

auto readFile(std::string_view path) -> std::string {
  // Read entire file into string
  // stolen from https://stackoverflow.com/a/116220
  constexpr auto read_size = std::size_t{4096};
  auto stream = std::ifstream{path.data()};
  if(!stream.is_open()){
    std::string errorMsg = std::string("Could not open file: ") + std::string(path);
    throw std::runtime_error(errorMsg);
  }
  stream.exceptions(std::ios_base::badbit);

  auto out = std::string{};
  auto buf = std::string(read_size, '\0');
  while (stream.read(& buf[0], read_size)) {
    out.append(buf, 0, stream.gcount());
  }
  out.append(buf, 0, stream.gcount());
  return out;
}

cl::Program buildProgramFromString(const std::string& source) {
  // Compile kernel source into program
  cl::Program program(source, false);

  try {
    program.build();
  } catch (cl::Error& e) {
    if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
      // Check the build status
      cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(cl::Device::getDefault());

      // Get the build log
      std::string bl = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl::Device::getDefault());
      std::cerr << bl << std::endl;
    } else {
      throw e;
    }
  }

  return program;
}

cl::Program buildProgramFromFile(const std::string& filename) {
  return buildProgramFromString(readFile(filename));
}
