#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <iostream>

#include <ocl_utility.hpp>

int main( int argc, char* argv[] )
{
  Catch::Session session;

  std::string platform;

  using namespace Catch::Clara;
  auto cli 
    = session.cli() // Get Catch's composite command line parser
    | Opt( platform, "platform" )["--platform"]("OpenCL platform to target");

  session.cli( cli ); 

  // Let Catch (using Clara) parse the command line
  int returnCode = session.applyCommandLine( argc, argv );
  if( returnCode != 0 ) // Indicates a command line error
      return returnCode;

  std::cout << "Running tests on platform: " << platform << std::endl;

  setDefaultPlatform(platform);

  return session.run();
}

