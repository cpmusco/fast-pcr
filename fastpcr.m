function x = fastpcr(A, b, lambda, iter, solver, method, tol)
%--------------------------------------------------------------------------
% Fast Matrix Polynomial Algorithm for Principal Component Regression
%
% usage : 
%
%  input:
%  * A : design matrix
%  * b : response vector
%  * lambda : eigenvalue cut off, default = ||A||_2^2/100.
%       All principal components of A'*A with eigenvalue < lambda will be 
%       ignored for the regression.
%  * iter : number of iterations, default = 10.
%       Each iteration requires the solution of one ridge regression
%       problem on A with ridge parameter lambda.
%  * solver: black box routine for ridge regression.
%       'CG' for Conjugate Gradient solver, default
%       'SVRG' for Stochastic Variance Reduced Gradient solver
%       ** or any other solver implemented in ridgeInv.m **
%  * method: the technique used for applying matrix polynomials.
%       'LANCZOS' for the standard Lanczos/Krylov subspace method, default
%       'EXPLICIT' for the explicit method analyzed in "Principal Component 
%       Projection Without Principal Component Analysis", Frostig et al. ICML '16
%  * tol: accuracy for calls to ridge regression, default 1e-5

%
%  output:
%  * x : approximate solution to PCR with parameter lambda. 
%        Specifically, let P_lambda be a matrix that projects onto the space
%        spanned by all eigenvectors of A'*A with eigenvalue > lambda. x
%        approximates P_lambda*pinv(A'*A)*A'*b.
%--------------------------------------------------------------------------

% Check input arguments and set defaults.
if nargin > 7
    error('fastpcr:TooManyInputs','requires at most 7 input arguments');
end
if nargin < 2
    error('fastpcr:TooFewInputs','requires at least 2 input arguments');
end
if nargin < 3
    lambda = svds(A,1)^2/100;
end
if nargin < 4
    iter = 50;
end
if nargin < 5
    solver = 'CG';
end
if nargin < 6
    method = 'LANCZOS';
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
L = svds(A,1)^2;

if(strcmp(method,'EXPLICIT'))
    pz = ridgeInv(A,A'*(A*z),lambda,solver,tol,L);

    % main polynomial recurrence (equivalent but slightly different than 
    % Frostig et al.)
    w = pz - z/2;
    for i = 1:iter
        t = ridgeInv(A,A'*(A*w),lambda,solver,tol,L);
        w = 4*(2*i+1)/(2*i)*ridgeInv(A,A'*(A*(w - t)),lambda,solver,tol,L);
        pz = pz + 1/(2*i+1)*w;
    end

elseif(strcmp(method,'LANCZOS'))
    pz = lanczos(@(g) ridgeInv(A,A'*(A*g),lambda,solver,tol,L), z, @(h) softStep(h,iter^2), iter);
    
else
    error('fastpcr:BadInput','the specified polynomial approximation method was not recognized')
end

%%% Principal Component Regression %%%
x = robustReg(A, pz, lambda, solver, tol, 'FULL');
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
s = s*2-1;
% in case s falls outside the expected range
s = min(s,1); s = max(s,-1);

% for speed we actually use a very near continuous approximation based on 
% the Gaussian CDF
step = normcdf(s,0,1/sqrt(2*q));
end