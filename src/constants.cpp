#include <constants.hpp>

Constants::Constants():
  nx{10},
  ny{10},
  ng{1}
{
  dx = 1.0/(nx+1);
  dy = 1.0/(ny+1);
}
