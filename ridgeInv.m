function [x] = ridgeInv(A, b, lambda, solver, tol)
%--------------------------------------------------------------------------
% Ridge Regression
%
% usage : 
%
%  input:
%  * A : n x d matrix
%  * b : d x 1 vector
%  * lambda : regularization parameter
%  * solver : 'CG' for iterative Conjugate Gradient solver, default
%             'SVRG' for iterative Stochastic Variance Reduced Gradient solver 
%  * tol : desired tolerance, default 1e-6
%
%  output:
%  * x : approximate solution to xstar = inv(A^T A + lambda*I)*b. 

%  x should satisfies ||(A^TA + lambda*I)x - b|| <= tol*||b||. 
%--------------------------------------------------------------------------

% Check input arguments and set defaults.
if nargin > 6
    error('ridgeInv:TooManyInputs','requires at most 5 input arguments');
end
if nargin < 3
    error('ridgeInv:TooFewInputs','requires at least 4 input arguments');
end
if nargin < 4
    solver = 'CG';
end
if nargin < 5
    tol = 1e-6;
end
if(lambda < 0 || tol < 0)
    error('ridgeInv:BadInput','one or more inputs outside required range');
end

function y = afun(z,~)
    y = A'*(A*z) + lambda*z;
end

if(strcmp(solver,'CG'))
% default MATLAB CG
    [x,~] = pcg(@afun,b,tol,1000);

elseif(strcmp(solver,'SVRG'))
% Simple (and slow) SVRG implementation (see Johnson, Zhang "Accelerating 
% Stochastic Gradient Descent using Predictive Variance Reduction"
    n = size(A,1);
    m = n/10;
    eta = 1/norm(A,'fro')^2;
    x = zeros(size(b));
    xt = x;
    done = false;
    g = (afun(x) - b)/n;
    while(~done)
        for j=1:m
            ind = randi(n);
            w = xt - x;
            xt = xt - eta*((A(ind,:)*w)*A(ind,:)' + lambda*w+ g);
        end
        xold = x;
        x = xt;
        if(norm(afun(x) - b) <= tol*norm(b))
            done = true;
        end
        gold = g;
        g = (afun(x) - b)/n;
        % Step size adjustment from "Barzilai-Borwein Step Size for 
        % Stochastic Gradient Descent", Tan, Ma, Dai, Qian
        eta = (1/m)*norm(x - xold)^2/((x - xold)'*(g - gold));
    end

%elseif(strcmp(solver,'MYSOLVER'))
% INSERT CODE FOR YOUR FAVORITE SOLVER HERE 

else
    error('ridgeInv:BadInput','the specificed solver was not recognized')
end


end
