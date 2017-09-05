function [x] = robustReg(A, p, lambda, solver, tol, method)
%--------------------------------------------------------------------------
% Robust Inversion (to map approximate Projection --> Regression)
%
% usage : 
%
%  input:
%  * A : matrix
%  * p : a vector spanned almost entirely by the top eigenvectors of A^TA,
%       specifically by eigenvectors with eigenvalue >= lambda
%  * lambda : eigenvalue cut off 
%  * tol : should be 0 if p is spanned *exactly* by A's top eigenvectors.
%       Otherwise it should be set to ||p - P_lambda p||/||p||, where
%       P_lambda is a projection operator onto A's top eigenvectors. In
%       practice it is not essential to compute tol exactly. Default = 1e-5.
%  * solver : solver to use in required calls to ridge regression oracle
%       See ridgeInv.m for potential options.
%  * method : 'SIMPLE' -- simple regularized regression, default (and suggested)
%             'FULL' -- matrix polynomial method used in Frostig et al (better
%
%  output:
%  * x : approximate solution to inv(P_lambda*A'*A)*p
%--------------------------------------------------------------------------

% Check input arguments and set defaults.
if nargin > 6
    error('robustReg:TooManyInputs','requires at most 4 input arguments');
end
if nargin < 4
    error('robustReg:TooFewInputs','requires at least 3 input arguments');
end
if nargin < 5
    tol = 1e-5;
end
if nargin < 6
    method = 'SIMPLE';
end

if(strcmp(method,'SIMPLE'))
    x = ridgeInv(A, p, sqrt(tol)*lambda, solver, sqrt(tol));

elseif(strcmp(method,'FULL'))
    riter = 2*log(1/tol);
    t = ridgeInv(A, p, lambda, solver, tol);
    x = t;
    for j = 1:riter-1
        x = t + lambda*ridgeInv(A, x, lambda, solver, tol);
    end
    
else
    error('robustReg:BadInput','the specificed method was not recognized')
end

end
