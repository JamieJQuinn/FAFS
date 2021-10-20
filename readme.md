# FPGA-Accelerated Fluid Solver (FAFS)

> to faff about - to dither or fuss

FAFS is a fast, eco-friendly, entirely (and delightfully) implicit fluids solver, based on Jos Stam's [Real-Time Fluid Dynamics for Games](https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf). The code solves the 2D incompressible Navier-Stokes equations using operator splitting, solving an advection step, where a quantity is advected from its previous location, a viscous diffusion step, where a matrix-free iterative method solves an implicit diffusion equation, and a projection step, where a Poisson equation is again solved iteratively to remove compressibility from the updated velocity field through pressure defined on a staggered grid. Additional force terms can also be included and Dirichlet, von Neumann and periodic boundary conditions are trivial to implement. The entirely implicit nature of the algorithm provides absolute stability, regardless of the timestep, at the cost of reduced accuracy and greatly enhanced numerical diffusion. No need to faff about with CFL numbers here!

The code is written in a kernel-based way, that is each step in the computation is defined as a kernel which acts on every grid point in one or many arrays. This approach can be more easily parallelised on heterogeneous platforms, or on specific hardware like FPGAs. Currently, FAFS is only parallelised on multi-core CPUs using OpenMP.

## Installation

### Faffing with Conan

While some dependencies are handled by conan, you must have conan and cmake already installed on your system. This is an unavoidable faff.

FAFS uses hdf5 as its main file format. This and its dependencies are easiest installed via the provided `conanfile.txt`:

```
conan install .. --build=hdf5 --build=zlib
```

### Faffing with Cmake

After the dependencies are installed, FAFS can be built with cmake:

```
mkdir build
cd build
cmake ..
make
```

## The faff of running FAFS

All initial conditions, boundary conditions, parameters, and anything else you might want to change are all hard-coded in FAFS, making running FAFS a faff. This is because JSON is a small faff in C++. I recommend faffing with the parameters defined in `src/constants.cpp` and running (from the `build` directory):

```
make && time OMP_NUM_THREADS=3 ./exe
```

This will run FAFS with 3 OpenMP threads which, on my machine, leaves me with one core while FAFS is running. This is not enough to run the faff that is Microsoft teams so I will be entirely unavailable until FAFS is complete.

Running FAFS as is will run a standard computational fluids test case, lid-driven cavity flow, at a Reynolds number of 10, grid points per side of 64, a timestep of 0.001 to a final time of 0.7. With 3 threads this should take around 5 seconds. YMMV.
