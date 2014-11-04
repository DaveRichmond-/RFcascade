function [x_aligned,A,b] = align_model_w_annotation(x,xp);

%

%
mean_x = mean(x,2);
x = x - repmat(mean_x, [1,size(x,2)]);

mean_xp = mean(xp,2);
xp = xp - repmat(mean_xp, [1,size(xp,2)]);

% vectorize
x_vec = reshape(x', [prod(size(x)),1]);
xp_vec = reshape(xp', [prod(size(xp)),1]);

if isequal(x_vec,xp_vec),
    
    A = [1, 0; 0, 1];
    
else,
    
    % calculate params for similarity transformation
    p1 = dot(x_vec,xp_vec)/dot(x_vec,x_vec);
    p2 = (dot(x(1,:),xp(2,:)) - dot(xp(1,:),x(2,:)))/dot(x_vec,x_vec);
    A = [p1, -p2; p2, p1];
        
end

% correct for offset of mean
x = x + repmat(mean_x, [1,size(x,2)]);
xp = xp + repmat(mean_xp, [1,size(xp,2)]);
b = mean_xp - (A + eye(2)) * mean_x;

% check result!
x_aligned = A*x + repmat(b,[1,size(x,2)]);