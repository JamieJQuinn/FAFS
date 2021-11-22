#include <constants.hpp>

Constants::Constants():
  nx{256},
  ny{256},
  ng{1},
  dt{0.001},
  totalTime{0.7},
  Re{100}
{
  dx = 1.0/(nx+1);
  dy = 1.0/(ny+1);
}
