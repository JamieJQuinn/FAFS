#pragma once

// User functions
void runJacobiIteration(OpenCLArray& out, OpenCLArray& initialGuess, OpenCLArray& temp, const real alpha, const real beta, const real gamma, const OpenCLArray& b, const int iterations = 20);
void calcDiffusionTerm(OpenCLArray& out, const OpenCLArray& f, const real dx, const real dy, const real Re);
void calcAdvectionTerm(OpenCLArray& out, const OpenCLArray& f, const OpenCLArray& vx, const OpenCLArray& vy, const real dx, const real dy);
void advanceEuler(OpenCLArray& out, const OpenCLArray& ddt, const real dt);

void applyVonNeumannBC(OpenCLArray& out);
void applyVonNeumannBC_x(OpenCLArray& out);
void applyVonNeumannBC_y(OpenCLArray& out);
void applyNoSlipBC(OpenCLArray& var);

void calcDivergence(OpenCLArray& out, OpenCLArray& fx, OpenCLArray& fy, const real dx, const real dy);
void applyProjectionX(OpenCLArray& out, OpenCLArray& f, const real dx);
void applyProjectionY(OpenCLArray& out, OpenCLArray& f, const real dy);

void advectImplicit(OpenCLArray& out, OpenCLArray& f, OpenCLArray& vx, OpenCLArray& vy, const real dx, const real dy, const real dt);
