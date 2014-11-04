function [xbar, R, Psi, Lambda, PsiT] = make_shape_model(table);

% parse data table into shape vectors

points = unique(table(:,1));
numpoints = size(points, 1);

slices = unique([table(:,4), table(:,7)],'rows');
numslices = size(slices, 1);

if (numslices*numpoints ~= size(table,1)),
    error('data not parsed correctly'),
end

xmat = zeros(numpoints, 2, numslices);              % 2 is the dimensionality of the image

for i=1:numslices,

    xmat(:,:,i) = table((i-1)*numpoints + 1 : i*numpoints, 2:3);

end


% center all shapes on origin

for i = 1:numslices,
    
    xmat(:,:,i) = xmat(:,:,i) - repmat(mean(xmat(:,:,i),1),[numpoints 1]);
    
end


% transfer to vector form (mix x and y pairs into single column)

xvec = zeros(numel(xmat(:,:,1)), numslices);

for i=1:numslices,
    
    xvec(:,i) = [xmat(:,1,i); xmat(:,2,i)];
    
end


% pick one example to be mean shape, and iteratively align all vectors with this reference
    
mean_indx = ceil(numslices/2);

[xvec, xbar, D] = normalize_shape_vectors(xvec, mean_indx);


% PCA

[mean_x, R, Psi, Lambda, PsiT] = myPCA(xvec);

%{
s = numslices;
n = numpoints*2;            % should be = size(xvec,1)

R = xvec - repmat(xbar, [1 s]);
Sigma = (R*R')/(s-1);

[Psi, Lambda, V] = svd(Sigma);
%}