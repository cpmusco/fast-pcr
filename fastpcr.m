function [x,pz] = fastpcr(A, b, lambda, iter, solver, method, tol)
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
%  * iter : number of iterations, default = 10
%       Each iteration requires the solution of one ridge regression
%       problem on A with ridge parameter lambda.
%  * solver: black box routine for ridge regression
%       'CG' for iterative Conjugate Gradient solver, default
%       'SVRG' for iterative Stochastic Variance Reduced Gradient solver
%       ** or any other solver implemented in ridgeInv.m **
%  * method: the technique used for applying matrix polynomials
%       'KRYLOV' for a standard Krylov subspace method, default
%       'EXP' for the explicit method analyzed in "Principal Component 
%       Projection Without Principal Component Analysis", Frostig et al.
%  * tol: accuracy for calls to ridge regression, default 1e-5
%
%
%  output:
%  * x : approximate solution to PCR with parameter lambda. 
%        Specifically, let P_lambda be a matrix that projects onto the
%        space spanned by all
%  eigenvectors of A'*A with eigenvalue > lambda. x approximates
%--------------------------------------------------------------------------

% Check input arguments and set defaults.
if nargin > 7
    error('fastpcr:TooManyInputs','requires at most 4 input arguments');
end
if nargin < 3
    error('fastpcr:TooFewInputs','requires at least 3 input arguments');
end
if nargin < 4
    iter = 10;
end
if nargin < 5
    solver = 'CG';
end
if nargin < 6
    method = 'KRYLOV';
end
if nargin < 7
    tol = 1e-5;
end
if(lambda < 0 || iter < 1)
    error('fastpcr:BadInput','one or more inputs outside required range');
end

%%% Principal Component Projection %%%

% for ridge regression, we project A'*b onto A's top singular directions 
% note however that the following code works for projecting any vector z
z = A'*b;

if(strcmp(method,'EXP'))
    pz = ridgeInv(A,A'*(A*z),lambda,solver,tol);

    % main polynomial recurrence (equivalent but slightly different than 
    % Frostig et al.)
    w = pz - z/2;
    for i = 1:iter
        t = ridgeInv(A,A'*(A*w),lambda,solver,tol);
        w = 4*(2*i+1)/(2*i)*ridgeInv(A,A'*(A*(w - t)),lambda,solver,tol);
        pz = pz + 1/(2*i+1)*w;
    end

elseif(strcmp(method,'KRYLOV'))
    pz = lanczos(@(g) ridgeInv(A,A'*(A*g),lambda,solver,tol), z, @(h) softStep(h,iter^2), iter);
    
else
    error('fastpcr:BadInput','the specificed method was not recognized')
end

%%% Principal Component Regression %%%
x = robustReg(A, pz, lambda, solver, tol, 'SIMPLE');
end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Soft Step Function
%--------------------------------------------------------------------------
function step = softStep(s,q)
% applies "soft step" function with parameter q to s, which should lie 
% in [0,1] (see https://arxiv.org/pdf/1708.07788.pdf, Equation 60) 
% if s < 1/2 it is mapped towards 0, if s > 1/2 it is mapped towards 1

% shift from [0,1] --> [-1,1]
step = s*2-1;
% in case s falls outside the expected range
step = min(step,1); step = max(step,-1);

if(step < 1/q) step = -1; end
if(step > 1/q) step = 1; end
% weight = 1;
% step = s;
% for i = 1:q
%     weight = weight*(2*i - 1)/(2*i);
%     step = step + weight*s.*(1-s.^2).^i;
% end
% shift from [-1,1] -->  [0,1]
step = (step+1)/2;
end



