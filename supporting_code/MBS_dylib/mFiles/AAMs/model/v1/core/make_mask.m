function [mask] = make_mask(X, Y, xbar_vector, SFPpoints, allPoints, varargin)

%

if nargin >= 6
    dilRadius = varargin{1};
    openRadius = varargin{2};
    L = varargin{3};
else
    dilRadius = 14;
    openRadius = 8;     % for SFP of whole somite
    L = 0.05;           % for SFP of forming boundary    
end



% some useful things

numpoints = length(allPoints);
xbar_mat = [xbar_vector(1:numpoints), xbar_vector(numpoints+1:2*numpoints)];
R90 =  [0, -1; 1, 0];
Rn90 = [0, 1; -1, 0];

% build mask, depending on points used in polygon

if isequal(SFPpoints, allPoints),
    
    xv = xbar_mat(:,1);
    yv = xbar_mat(:,2);

    mask = inpolygon(X, Y, [xv; xv(1)], [yv; yv(1)]);
    mask = reshape(mask, size(X));
    
    % open and dilate mask
    
    nhood = fspecial('disk', openRadius);
    nhood = ceil(nhood);
    mask = imopen(mask,nhood);
    
    nhood = fspecial('disk', dilRadius);
    nhood = ceil(nhood);
    mask = imdilate(mask,nhood);
    
    clear nhood
    
elseif isequal(SFPpoints, [2;3;4]),
    
    % define 'useful' vectors
    
    p2 = xbar_mat(3,:);
    p3 = xbar_mat(4,:);
    p4 = xbar_mat(5,:);
    
    % calculate normed vectors along axes of boundary
    
    n2 = (p2 - p3)' / norm(p2 - p3);
    n4 = (p4 - p3)' / norm(p4 - p3);
    n3 = (n2 + n4) / norm(n2 + n4);
    
    %
    
    v = zeros(8,2);
    v(1,:) = p2 + L*(R90*n2)';
    v(2,:) = p2 + L*n2';
    v(3,:) = p2 + L*(Rn90*n2)';
    
    v(5,:) = p4 + L*(R90*n4)';
    v(6,:) = p4 + L*n4';
    v(7,:) = p4 + L*(Rn90*n4)';
    
    v(4,:) = p3 - L*n3';
    v(8,:) = p3 + L*n3';
    
    xv = v(:,1);
    yv = v(:,2);
    
    mask = inpolygon(X, Y, [xv; xv(1)], [yv; yv(1)]);
    mask = reshape(mask, size(X));
    
elseif isequal(SFPpoints, [2;4]),
    
    % define 'useful' vectors
    
    p2 = xbar_mat(3,:);
    p4 = xbar_mat(5,:);
    
    % calculate normed vectors along axes of boundary
    
    n2 = (p2 - p4)' / norm(p2 - p4);
    n4 = -n2;
    
    %
    
    v = zeros(6,2);
    v(1,:) = p2 + L*(R90*n2)';
    v(2,:) = p2 + L*n2';
    v(3,:) = p2 + L*(Rn90*n2)';
    
    v(4,:) = p4 + L*(R90*n4)';
    v(5,:) = p4 + L*n4';
    v(6,:) = p4 + L*(Rn90*n4)';
        
    xv = v(:,1);
    yv = v(:,2);
    
    mask = inpolygon(X, Y, [xv; xv(1)], [yv; yv(1)]);
    mask = reshape(mask, size(X));
    
    
elseif isequal(SFPpoints, [0,[2:7]]'),
    
    xv = xbar_mat(SFPpoints+1,1);
    yv = xbar_mat(SFPpoints+1,2);

    mask = inpolygon(X, Y, [xv; xv(1)], [yv; yv(1)]);
    mask = reshape(mask, size(X));
    
    % open and dilate mask
    
    nhood = fspecial('disk', openRadius);
    nhood = ceil(nhood);
    mask = imopen(mask,nhood);
    
    nhood = fspecial('disk', dilRadius);
    nhood = ceil(nhood);
    mask = imdilate(mask,nhood);
    
else,
    
    error('points to include in SFP doesnt correspond to allowed set');    

end