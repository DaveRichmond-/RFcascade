function [interp_value] = bilinear_interp(image, x, y);

% nearest grid points around interp

x1 = floor(x);
x2 = ceil(x);
dx = x2 - x1;

y1 = floor(y);
y2 = ceil(y);
dy = y2 - y1;

Q11 = [x1, y1];
Q12 = [x1, y2];
Q21 = [x2, y1];
Q22 = [x2, y2];

if (dx==0) && (dy~=0),

    interp_value = NaN;     %fix this later
    
elseif (dx~=0) && (dy==0),
    
    interp_value = NaN;     % fix this later
    
elseif (dx==0) && (dy==0),
    
	interp_value = image(x1,y1);
    
else
    
    interp_value = (1/(dx*dy)) * ...
        (image(Q11(1), Q11(2))*(x2-x)*(y2-y) + ...
        image(Q21(1), Q21(2))*(x-x1)*(y2-y) + ...
        image(Q12(1), Q12(2))*(x2-x)*(y-y1) + ...
        image(Q22(1), Q22(2))*(x-x1)*(y-y1));
    
end