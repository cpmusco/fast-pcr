function [x,patb] = fastpcr(A, b, lambda, iter, solver, method)
%--------------------------------------------------------------------------
% Fast Matrix Polynomial Algorithm for Principal Component Regression
%
% usage : 
%
%  input:
%  * A : design matrix
%  * b : response vector
%  * lambda : eigenvalue cut off 
%       All principal components of A'*A with eigenvalue < lambda will be 
%       ignored for the regression.
%  * iter : number of iterations
%       Each iteration requires the solution of one ridge regression
%       problem on A with ridge parameter lambda, default = 25
%  * solver: black box routine for ridge regression
%       'CG' for iterative Conjugate Gradient solver, default
%       'SVRG' for iterative Stochastic Variance Reduced Gradient solver
%       ** or any other solver implemented in ridgeInv.m **
%  * method: the technique used for applying matrix polynomials
%       'KRYLOV' for a standard Krylov subspace method, default
%       'EXP' for the explicit method analyzed in "Principal Component 
%       Projection Without Principal Component Analysis", Frostig et al.
%       'ACCEL' for the acclerated explicit method analyzed in "Principal Component 
%       Projection Without Principal Component Analysis", Frostig et al.
%
%
%  output:
%  * patb : approximation projection of A^T b onto top subspace of A
%  * x : approximate solution to PCR
%--------------------------------------------------------------------------

% Check input arguments and set defaults.
if nargin > 4
    error('rpcr:TooManyInputs','requires at most 4 input arguments');
end
if nargin < 3
    error('rpcr:TooFewInputs','requires at least 3 input arguments');
end
if nargin < 4
    iter = 25;
end
if(lambda < 0 || iter < 1)
    error('fpcr:BadInput','one or more inputs outside required range');
end

%%% Principal Component Projection %%%

% for ridge regression, we project A'*b onto A's top singular directions 
% note however that the following code works for projecting any vector z
z = A'*b;

% start building up our projected vector, pz
pz = ridgeReg(A,A*z,lambda);

% main polynomial recurrence (equivalent but slightly different than 
% http://arxiv.org/abs/1602.06872)
w = pz - z/2;
for i = 1:iter
    w = 4*(2*i+1)/(2*i)*ridgeReg(A,A*(w - ridgeReg(A,A*w,lambda)), lambda);
    pz = pz + 1/(2*i+1)*w;
end
patb = pz;

%%% Principal Component Regression %%%
x = robustReg2(A,pz,lambda);
end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Ridge Regression
%
%  note: MATLAB's lsqr is used as default, but could be replaced with any
%  fast ridge regression routine
%--------------------------------------------------------------------------
function x = ridgeReg(A,b,lambda)
    wth = size(A,2);
    [x,~] = lsqr([A;sqrt(lambda)*eye(wth)],[b;zeros(wth,1)]);
end

%--------------------------------------------------------------------------
% Robust Inversion (to map Projection --> Regression)
%--------------------------------------------------------------------------

function x = robustReg1(A,pz,lambda)
% method used in http://arxiv.org/abs/1602.06872
    riter = 20; % default
    function y = afun(z,~)
        y = A'*(A*z) + lambda*z;
    end
    [t,~] = lsqr(@afun,pz);
    x = t;
    for j = 1:riter-1
        [u,~] = lsqr(@afun,x);
        x = t + lambda*u;
    end
end

function x = robustReg2(A,pz,lambda)
% simpler method that works well in practice
    tol = 1e-5; %default
    function y = afun(z,~)
        y = A'*(A*z) + tol*lambda*z;
    end
    [x,~] = pcg(@afun,pz);
end




