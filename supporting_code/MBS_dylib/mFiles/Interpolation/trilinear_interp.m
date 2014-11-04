function [Ic] = trilinear_interp(I, x, y, z);

% interpolates the function V, to the non-integer indices x,y,z


% round up/down to nearest indices around interp

x0 = floor(x);
x1 = ceil(x);
xd = x - x0;

y0 = floor(y);
y1 = ceil(y);
yd = y - y0;

z0 = floor(z);
z1 = ceil(z);
zd = z - z0;

% 

c00 = I(x0, y0, z0)*(1 - xd) + I(x1, y0, z0)*xd;
c10 = I(x0, y1, z0)*(1 - xd) + I(x1, y1, z0)*xd;
c01 = I(x0, y0, z1)*(1 - xd) + I(x1, y0, z1)*xd;
c11 = I(x0, y1, z1)*(1 - xd) + I(x1, y1, z1)*xd;

c0 = c00*(1-yd) + c10*yd;
c1 = c01*(1-yd) + c11*yd;

c = c0*(1-zd) + c1*zd;

%

Ic = c;