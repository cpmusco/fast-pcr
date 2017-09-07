# fast-pcr -- Fast Iterative Principal Component Regression

This MATLAB code implements iterative methods for [Principal Component Regression](https://en.wikipedia.org/wiki/Principal_component_regression) based on the work in [Principal Component Projection Without Principal Component Analysis](http://proceedings.mlr.press/v48/frostig16.html).

## Installataion

Download `fastpcr.m`,`lanczos.m`,`ridgeInv.m`, and `robustReg.m`, [add to MATLAB path](https://www.mathworks.com/help/matlab/ref/addpath.html), or include directly in project directory.

## Documentation

**Principal Component Regression (PCR)** is a common and effective form of regularized linear regression. Given an n x d matrix **A** and threshold &lambda;, let **P** be a d x d projection matrix onto the span of all right singular vectors of **A** with corresponding singular value > &lambda;.  PCR solves **x** = **P**(**A**<sup>T</sup>**A**)<sup>-1</sup>**A**<sup>T</sup>**b**. In other words, it solves a standard linear regression problem but restricts the solution to lie in the space spanned by **A**'s top singular vectors.
