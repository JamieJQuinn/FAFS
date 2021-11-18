#pragma once

// User functions
void runJacobiIteration(OpenCLArray& out, OpenCLArray& in, const real alpha, const real beta, const OpenCLArray& b, const int iterations=20);
void calcDiffusionTerm(OpenCLArray& out, const OpenCLArray& f, const real dx, const real dy, const real Re);
void calcAdvectionTerm(OpenCLArray& out, const OpenCLArray& f, const OpenCLArray& vx, const OpenCLArray& vy, const real dx, const real dy);
void advanceEuler(OpenCLArray& out, const OpenCLArray& ddt, const real dt);

