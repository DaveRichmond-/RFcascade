function [H] = LK_Hessian_Robust(SD, SD_rho_p);

%

H = SD'*SD_rho_p;