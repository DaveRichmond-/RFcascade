function [interp_value] = bilinear_interp_vec(image, x, y);

% nearest grid points around interp
x = x(:);
x1 = floor(x);
x2 = ceil(x);
dx = x2 - x1;

y = y(:);
y1 = floor(y);
y2 = ceil(y);
dy = y2 - y1;

Q11 = [x1, y1];
Q12 = [x1, y2];
Q21 = [x2, y1];
Q22 = [x2, y2];

idxQ11 = sub2ind(size(image), Q11(:,1), Q11(:,2));
idxQ12 = sub2ind(size(image), Q12(:,1), Q12(:,2));
idxQ21 = sub2ind(size(image), Q21(:,1), Q21(:,2));
idxQ22 = sub2ind(size(image), Q22(:,1), Q22(:,2));

% if (dx==0) && (dy~=0),
% 
%     interp_value = NaN;     %fix this later
%     
% elseif (dx~=0) && (dy==0),
%     
%     interp_value = NaN;     % fix this later
%     
% elseif (dx==0) && (dy==0),
%     
% 	interp_value = image(x1,y1);
%     
% else

% note: no safety checks.  if dx = 0 or dy = 0, will fail.
% interp_value = (1./(dx.*dy)) .* (image(Q11(:,1), Q11(:,2)).*(x2-x).*(y2-y) + image(Q21(:,1), Q21(:,2)).*(x-x1).*(y2-y) + image(Q12(:,1), Q12(:,2)).*(x2-x).*(y-y1) + image(Q22(:,1), Q22(:,2)).*(x-x1).*(y-y1));

interp_value = (1./(dx.*dy)) .* (image(idxQ11).*(x2-x).*(y2-y) + image(idxQ21).*(x-x1).*(y2-y) + image(idxQ12).*(x2-x).*(y-y1) + image(idxQ22).*(x-x1).*(y-y1));

% end