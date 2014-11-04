function [rho_p] = robust_error(Esq,sigma);

% evaluate the robust error function, based on Baker, et al., (LK Part II) Eq'n 51
% This is known as the Huber Robust Error Function, which after a change of variables is:
% rho(t) = {(1/2)*t;                            0 <= t <= sigma^2
%          {sigma*sqrt(t) - 1/2 * sigma^2;      t > sigma^2

% I'm interested in evaluating the derivative of this function, rho_p
% rho_p(t) = {1/2;                              0 <= t <= sigma^2
%            {(1/2)*sigma/sqrt(t);              t > sigma^2

% initialize rho_p
rho_p = zeros(length(Esq),1);

% find pixels below the threshold, BT, and above threshold, AT
BT = (Esq <= sigma^2);
AT = 1 - BT;

% return indices of above
indx_BT = find(BT);
indx_AT = find(AT);

% evaluate derivative of robust error fnct
rho_p(indx_BT) = 1/2;
rho_p(indx_AT) = (1/2)*sigma*(1./sqrt(Esq(indx_AT)));