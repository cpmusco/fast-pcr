# fast-pcr -- Fast Iterative Principal Component Regression

This MATLAB code implements iterative methods for [Principal Component Regression](https://en.wikipedia.org/wiki/Principal_component_regression) based on the work in [Principal Component Projection Without Principal Component Analysis](http://proceedings.mlr.press/v48/frostig16.html).

## Installation

Download `fastpcr.m`,`lanczos.m`,`ridgeInv.m`, and `robustReg.m`, [add to MATLAB path](https://www.mathworks.com/help/matlab/ref/addpath.html), or include directly in project directory.

## Documentation

**Principal Component Regression (PCR)** is a common and effective form of regularized linear regression. Given an n x d matrix **A** and threshold &lambda;, let **P**<sub>&lambda;</sub> be a d x d projection matrix onto the span of all right singular vectors (i.e. principal components) of **A** with corresponding squared singular value > &lambda;.  PCR computes **x** = **P**<sub>&lambda;</sub>(**A**<sup>T</sup>**A**)<sup>-1</sup>**A**<sup>T</sup>**b**. In other words, it solves a standard linear regression problem but restricts the solution to lie in the space spanned by **A**'s top singular vectors.

Most implementations of PCR first perform an eigendecoposition of **A**<sup>T</sup>**A** in order to compute **P**<sub>&lambda;</sub> before seperately solving a regression problem. The eigendecomposition is a computational bottle neck. 
  
`fastpcr` avoids this step entirely through matrix polynomial methods (either explicit or implicit via the Lanczos method). To do so, it requires access to an algorithm for standard [&#8467;<sub>2</sub> regularized regression](https://en.wikipedia.org/wiki/Tikhonov_regularization). It uses a few calls to this algorithm to construct a solution to the Principal Component Regression problem.

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
A = randn(10000,4000); b = randn(10000,1);
[U,S,V] = svd(A,'econ');
% modify to have (slightly) decaying spectrum
k = 500;
s = diag(S); s(1:k) = s(1:k)+10;
A = U*diag(s)*V';
lambda = (s(k)-5)^2;

% compute approximate PCR solution
tic; x = fastpcr(A, b, lambda); toc;
Elapsed time is 10.399166 seconds.
```

`fastpcr` is typically faster than a direct PCR solve, but is still very accurate.

```
% compare to direct method
tic;
[V,D] = eig(A'*A); 
xDirect = V*(D>lambda)*V'*(A\b);
toc;
Elapsed time is 37.365212 seconds.

% parameter vector error
norm(x-xDirect)/norm(xDirect)
0.0092

% projection error
norm(x - V*(D>lambda)*V'*x)/norm(x)
0.0048
````

### Implementation Options and Parameter Tuning

If a higher accuracy solution is required, the `iter` and `tol` parameters should be increased from the defaults of `40` and `1e-3`:
```
x = fastpcr(A, b, lambda, 200, 'CG', 'LANCZOS', 1e-8);
norm(x-xDirect)/norm(xDirect)
1.0304e-08
```

Doing so will slow the algorithm as the number of calls to a ridge regression algorithm is equal to `iter` and the algorithm will be run to accuracy `tol`. The most effective way to improve the runtime of `fastpcr` is to provide a faster ridge regression algorithm in `ridgeInv.m`. This algorithm can be customized to your dataset and computational environment. 

We do not recommend running `fastpcr` with the `method` option set to `EXPLICIT`. Doing so almost always results in a less accurate solution. A full error analysis of the `LANCZOS` method can be found in [Stability of the Lanczos Method for Matrix Function Approximation](https://arxiv.org/abs/1708.07788).


