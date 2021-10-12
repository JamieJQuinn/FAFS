#include <constants.hpp>

Constants::Constants():
  nx{12},
  ny{12},
  ng{1}
{
  dx = 1.0/(nx+1);
  dy = 1.0/(ny+1);
}
