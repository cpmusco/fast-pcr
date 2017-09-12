# fast-pcr -- Fast Iterative Principal Component Regression

This MATLAB code implements iterative methods for [Principal Component Regression](https://en.wikipedia.org/wiki/Principal_component_regression) based on the work in [Principal Component Projection Without Principal Component Analysis](http://proceedings.mlr.press/v48/frostig16.html).

## Installation

Download `fastpcr.m`,`lanczos.m`,`ridgeInv.m`, and `robustReg.m`, [add to MATLAB path](https://www.mathworks.com/help/matlab/ref/addpath.html), or include directly in project directory.

## Documentation

**Principal Component Regression (PCR)** is a common and effective form of regularized linear regression. Given an n x d matrix **A** and threshold &lambda;, let **P**<sub>&lambda;</sub> be a d x d projection matrix onto the span of all right singular vectors (i.e. principal components) of **A** with corresponding squared singular value > &lambda;.  PCR computes **x** = **P**<sub>&lambda;</sub>(**A**<sup>T</sup>**A**)<sup>-1</sup>**A**<sup>T</sup>**b**. In other words, it solves a standard linear regression problem but restricts the solution to lie in the space spanned by **A**'s top singular vectors.

### Usage

**Input:**
`fastpcr(A, b, lambda, iter, solver, method, tol)`

- `A` : design matrix
-  `b` : response vector
- `lambda` : eigenvalue cut off, default = ||`A`<sup>T</sup>`A`||<sub>2</sub>/100. All eigenvectors of `A`<sup>T</sup>`A` with eigenvalue < `lambda` (i.e., all singular vectors of `A` with squared singular value < `lambda`) will be ignored for the regression.
- `iter` : number of iterations, default = 50. Each iteration requires the solution of one ridge regression problem on `A` with ridge parameter `lambda`.
- `solver`: black box routine for ridge regression, default = 'CG'. Set to 'CG' for Conjugate Gradient solver, 'SVRG' for Stochastic Variance Reduced Gradient solver, or any other solver implemented in `ridgeInv.m`.
- `method`: the technique used for applying matrix polynomials, default = 'LANCZOS'. Set to 'LANCZOS' for a standard Lanczos method analyzed [here](https://arxiv.org/abs/1708.07788), or 'EXPLICIT' for the explicit method analyzed in our [ICML paper](http://proceedings.mlr.press/v48/frostig16.html).
- `tol`: accuracy for calls to ridge regression, default 1e-5

**Output:**

- `x` : approximate solution to PCR with parameter lambda.

### Example

**Approximate Principal Component Regression on random dataset**

```
% generate random test problem
A = randn(4000,1000); b = randn(4000,1); k = 500;
[U,S,V] = svds(A,1000);
% modify to have (slightly) decaying spectrum
s = diag(S); s(1:k) = s(1:k)+4;
S = diag(s); A = U*S*V';
lambda = (s(k)-2)^2;

% compute approximate PCR solution
x = fastpcr(A, b, lambda, 50, 'CG', 'LANCZOS', 1e-5);
```

`fastpcr` can be much faster than a direct PCR solve for large inputs, but is still very accurate.

```
% compare to direct method
x = fastpcr(A, b, lambda, 50, 'CG', 'LANCZOS', 1e-5);
[~,~,V] = svds(A,k); xDirect = V(:,1:k)*(V(:,1:k)'*(A\b));

% parameter vector error
norm(x-xDirect)/norm(xDirect)
0.0064

% projection error
norm(x - V(:,1:k)*(V(:,1:k)'*x))/norm(x)
0.0060
````