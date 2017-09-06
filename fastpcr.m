function [x,patb] = fastpcr(A, b, lambda, iter, solver, method, tol)
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
%       'ACCEL' for the acclerated explicit method analyzed in "Principal Component 
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
if nargin < 6
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
    pz = ridgeReg(A,A*z,lambda,solver,tol);

    % main polynomial recurrence (equivalent but slightly different than 
    % Frostig et al.)
    w = pz - z/2;
    for i = 1:iter
        t = ridgeReg(A,A*z,lambda,solver,tol);
        w = 4*(2*i+1)/(2*i)*ridgeReg(A,A*(w - t),lambda,solver,tol);
        pz = pz + 1/(2*i+1)*w;
    end

elseif(strcmp(method,'KRYLOV'))
    % Allocate space for Krylov subspace.
    width = size(A,2);
    K = zeros(width,iter);

    % Construct Krylov subspace for the matrix inv(A'*A+lambda*I)*A'A
    K(:,1) = normc(A'*b);
    for i=2:iter
        K(:,i) = ridgeReg(A,A*K(:,i-1),lambda);
        for j = 1:i-1
            K(:,i) = normc(K(:,i) - K(:,j)*(K(:,j)'*K(:,i)));
        end
    end
    Q = K;
    % Krylov Postprocessing
    wth = size(Q,2);
    T = zeros(wth);
    for i = 1:wth
        % Compute Q'*(A'A+\lambda*I)^-1*A'A*Q
        T(:,i) = Q'*ridgeReg(A,A*Q(:,i),lambda);
    end
    % Economy size dense SVD.
    [U,S,V] = svd(T,0);
    % symmetric step applied to S
    s = diag(S);
    step = (s > .5);
    patb = Q*(U*(diag(step)*(V'*(Q'*(A'*b)))));
    
    
else
    error('fastpcr:BadInput','the specificed method was not recognized')
end

    x = ridgeInv(A, p, sqrt(tol)*lambda, solver, sqrt(tol));


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




