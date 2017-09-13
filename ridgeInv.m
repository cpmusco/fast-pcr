function [x] = ridgeInv(A, b, lambda, solver, tol, L)
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
%  * L : estimate for norm(A'*A), if not provided it will be computed
%
%  output:
%  * x : approximate solution to xstar = inv(A^T A + lambda*I)*b. 

%  x satisfies ||(A^TA + lambda*I)x - b|| <= tol*||b||. 
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
if nargin < 6
    % compute rough spectral norm estimate
    top = rand(size(A,2),1);
    for i=1:5
        top = A'*(A*top); top = normc(top);
    end
    L = (top'*A')*(A*top);
end
if(lambda < 0 || tol < 0 || L < 0)
    error('ridgeInv:BadInput','one or more inputs outside required range');
end

function y = afun(z,~)
    y = A'*(A*z) + lambda*z;
end

if(strcmp(solver,'CG'))
% default MATLAB CG
    [x,~] = pcg(@afun,b,tol,100 + ceil(sqrt(L/lambda)));

elseif(strcmp(solver,'SVRG'))
% Simple (and slow) SVRG implementation, sampling rows by their squared norms
    n = size(A,1);
    % SVRG variance bound
    S = sum(sum(A.^2)) + lambda;
    % Set step size and epoch length
    eta = 1/(2*S);
    m = ceil(S/lambda);
    % sampling probabilities proportional to sqaured row norms
    rowProbs = sum(A.^2,2)/sum(sum(A.^2));
    % initialize x = 0
    x = zeros(size(b));
    xt = x;
    done = false;
    % full gradient computation
    g = (afun(x) - b);
    while(~done)
        ind = randsample(n,m,true,rowProbs);
        for j=1:m
            % stochastic gradient update step
            w = xt - x;
            xt = xt - eta/rowProbs(ind(j))*(A(ind(j),:)'*(A(ind(j),:)*w)) - eta*lambda*w - eta*g;
        end
        x = xt;
        if(norm(afun(x) - b) <= tol*norm(b))
            done = true;
        end
        g = (afun(x) - b);
    end

%elseif(strcmp(solver,'MYSOLVER'))
% INSERT CODE FOR YOUR FAVORITE SOLVER HERE 

else
    error('ridgeInv:BadInput','the specificed solver was not recognized')
end


end

