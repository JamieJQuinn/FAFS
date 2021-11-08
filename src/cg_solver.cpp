#include <cg_solver.hpp>
#include <array2d.hpp>

//void CGSolver::solve(Array& out, kernelFn A, const Array& b) {
  //auto sqrKernel = [&](const Array& f, const int i, const int j) {
    //return f(i,j)*f(i,j);
  //}
  //// out also acts as initial guess
  //res.applyKernel([&](Array& f, const int i, const int j) {
      //f(i,j) = b(i,j) - A(out,i,j);
  //});
  //d = res; // copy to initial d
  //for(int i=0; i<nIterations; ++i) {
    //res.applyKernel(sqr, res2); 
    //resTres = res2.sum(); // find residual norm
    //d.applyKernel(A, p);

    //p.applyKernel(
    //dTp = 
    //dTp = sum(d*p);
    //alpha = resTres/pTp;
    //x += alpha*d; // Calc new pos
    //res += -alpha*p; // calc new residual
    //resTresPrev = resTres; // save old resNorm for beta calculation

    //res.applyKernel(sqr, res2); 
    //resTres = res2.sum(); // find residual norm
    //beta = resTres/resTresOld;
    //d = res + beta * d;
  //}
//}
