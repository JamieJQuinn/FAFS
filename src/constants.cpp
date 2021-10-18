#include <constants.hpp>

Constants::Constants():
  nx{64},
  ny{64},
  nu{0.1},
  totalTime{0.7},
  dt{0.001},
  ng{1}
{
  dx = 1.0/(nx+1);
  dy = 1.0/(ny+1);
}
