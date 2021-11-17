#include <cmath>

#include <openmp_kernels.hpp>

real calcAdvection(const Array& f, const int i, const int j, const real dx, const real dy, const real dt, const int nx, const int ny, const int ng, const Array& vx, const Array& vy) {
  // figure out where the current piece has come from (in index space)
  real x = i - dt*vx(i,j)/dx;
  real y = j - dt*vy(i,j)/dy;
  // clamp to int indices and ensure inside domain
  int rigIdx = nx-1+ng;
  int lefIdx = -ng;
  int topIdx = ny-1+ng;
  int botIdx = -ng;
  int x2 = clamp(int(std::floor(x+1)),rigIdx, lefIdx);
  int x1 = clamp(int(std::floor(x))  ,rigIdx, lefIdx);
  int y2 = clamp(int(std::floor(y+1)),topIdx, botIdx);
  int y1 = clamp(int(std::floor(y))  ,topIdx, botIdx);
  x = clamp(x, real(rigIdx), real(lefIdx));
  y = clamp(y, real(topIdx), real(botIdx));
  //std::cout << x << ", " << y << std::endl;
  //std::cout << x1 << ", " << x2 << ", " << y1 << ", " << y2 << std::endl;
  //std::cout << x1 << ", " << int(x) << std::endl;
  //// bilinearly interpolate
  real fy1, fy2;
  if(x1!=x2) {
    real x1Weight = (x2-x)/(x2-x1);
    real x2Weight = (x-x1)/(x2-x1);
    fy1 = x1Weight*f(x1, y1) + x2Weight*f(x2, y1);
    fy2 = x1Weight*f(x1, y2) + x2Weight*f(x2, y2);
  } else {
    fy1 = f(x1, y1);
    fy2 = f(x1, y2);
  }
  real fAv;
  if(y1!=y2) {
    real y1Weight = (y2-y)/(y2-y1);
    real y2Weight = (y-y1)/(y2-y1);
    fAv = y1Weight*fy1 + y2Weight*fy2;
  } else {
    fAv = fy1;
  }
  return fAv;
}

real calcJacobiStep(const Array& f, const real alpha, const real beta, const Array& b, const int i, const int j) {
  return (alpha*b(i,j) + f(i,j+1) + f(i,j-1) + f(i+1,j) + f(i-1,j))/beta;
}

void applyJacobiStep(Array& out, const Array& f, const real alpha, const real beta, const Array& b) {
#pragma omp parallel for collapse(2)
  for (int i=0; i<out.nx; ++i) {
    for(int j=0; j<out.ny; ++j) {
      out(i,j) = calcJacobiStep(f, alpha, beta, b, i, j);
    }
  }
}

void runJacobiIteration(Array& out, Array& in, const real alpha, const real beta, const Array& b, const int iterations) {
  for(int i=0; i<iterations; ++i) {
    applyJacobiStep(out, in, alpha, beta, b);
    in.swap(out);
  }
}

real ddx(const Array& f, const real dx, const int i, const int j) {
  return (f(i+1,j)-f(i-1,j))/(2.0f*dx);
}

real ddy(const Array& f, const real dy, const int i, const int j) {
  return (f(i,j+1) - f(i,j-1))/(2.0f*dy);
}

void advectImplicit(Array& out, const Array& f, const Array& vx, const Array& vy, const real dx, const real dy, const real dt, const int nx, const int ny, const int ng) {
#pragma omp parallel for collapse(2)
  for (int i=0; i<out.nx; ++i) {
    for(int j=0; j<out.ny; ++j) {
      out(i,j) = calcAdvection(f, i, j, dx, dy, dt, nx, ny, ng, vx, vy);
    }
  }
}

real calcAdvectionTerm(const Array& f, const Array& vx, const Array& vy, const real dx, const real dy, const int i, const int j) {
  return -(vx(i,j) * ddx(f, dx, i, j) + vy(i,j) * ddy(f, dy, i, j));
}

void calcAdvectionTerm(Array& out, const Array& f, const Array& vx, const Array& vy, const real dx, const real dy) {
#pragma omp parallel for collapse(2)
  for(int i=0; i<out.nx; ++i) {
    for(int j=0; j<out.ny; ++j) {
      out(i,j) = calcAdvectionTerm(f, vx, vy, dx, dy, i, j);
    }
  }
}

real calcDiffusionTerm(const Array& f, const real dx, const real dy, const int i, const int j) {
  return (f(i,j+1) + f(i,j-1) + f(i+1,j) + f(i-1,j) - 4*f(i,j))/(dx*dy);
}

void calcDiffusionTerm(Array& out, const Array& f, const real dx, const real dy) {
#pragma omp parallel for collapse(2)
  for(int i=0; i<out.nx; ++i) {
    for(int j=0; j<out.ny; ++j) {
      out(i,j) = calcDiffusionTerm(f, dx, dy, i, j);
    }
  }
}

void advanceEuler(Array& out, const Array& ddt, const real dt) {
#pragma omp parallel for collapse(2)
  for(int i=0; i<out.nx; ++i) {
    for(int j=0; j<out.ny; ++j) {
      out(i,j) = out(i,j) + ddt(i,j)*dt;
    }
  }
}

void calcDivergence(Array& out, const Array& fx, const Array& fy, const real dx, const real dy) {
#pragma omp parallel for collapse(2)
  for(int i=0; i<out.nx; ++i) {
    for(int j=0; j<out.ny; ++j) {
      out(i,j) = (fx(i,j) - fx(i-1,j))/dx + (fy(i,j) - fy(i,j-1))/dy;
    }
  }
}

void applyProjectionX(Array& out, const Array& f, const real dx) {
#pragma omp parallel for collapse(2)
  for(int i=0; i<out.nx; ++i) {
    for(int j=0; j<out.ny; ++j) {
      out(i,j) = out(i,j) - (f(i+1,j)-f(i,j))/dx;
    }
  }
}

void applyProjectionY(Array& out, const Array& f, const real dy) {
#pragma omp parallel for collapse(2)
  for(int i=0; i<out.nx; ++i) {
    for(int j=0; j<out.ny; ++j) {
      out(i,j) = out(i,j) - (f(i,j+1)-f(i,j))/dy;
    }
  }
}
