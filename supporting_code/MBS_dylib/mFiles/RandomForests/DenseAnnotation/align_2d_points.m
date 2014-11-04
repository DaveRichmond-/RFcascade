%% align somite landmark points

clear all

% create list of filenames
listing = dir(pwd);

k=1;
for i = 1:length(listing),   
    if length(strfind(listing(i).name,'.tif') >= 0),
        fname_list{k} = listing(i).name;
        k = k+1;
    end
end
clear i k listing

% load positional information
load('pos.mat'),

% load flip_indx, which indicates if the embryo is flipped across vertical axis
load('flipIndx.mat')

% load reference image
im1 = imread('Focused 120807_bf_f0000_frame103.tif');

% set the first embryo as the reference
xp = reshape(pos(:,:,:,1), [size(pos,1) size(pos,2)*size(pos,3)])';

% fix xp coords to be Cartesian Coords
xp(:,1) = xp(:,1) - 1;
xp(:,2) = size(im1,1) - xp(:,2);

% correct for possible flip
if flip_indx(1),
    xp(:,1) = size(im1,2) - xp(:,1) - 1;
end

for i = 2:size(pos,4),
    
    % read from table and reshape
    x  = reshape(pos(:,:,:,i), [size(pos,1) size(pos,2)*size(pos,3)])';
    
    % Cartesian Coords
    %x(:,1) = x(:,1) - 1;
    %x(:,2) = size(im1,1) - x(:,2);
    
    % flip
    if flip_indx(i),
        x(:,1) = size(im1,2) - x(:,1) - 1;
    end
    
    % rigid transform!!
    [R,T,Yf,Err] = rot3dfit([x, ones(size(x,1),1)], [xp, ones(size(xp,1),1)]);
    
    % rewrite transform compactly.  Note, A is transposed from what I would have expected.
    A = R;
    A(3,1) = T(1);
    A(3,2) = T(2);
    
    % nonsense
    %{
    % flip rotation direction
    A(1,2) = -A(1,2);
    A(2,1) = -A(2,1);
    %}
    
    % apply to corresponding images
    im2 = imread(fname_list{i});
    tform = maketform('affine',A);
    im2_trans = imtransform(im2,tform,'nearest','XData',[1 size(im2,2)],'YData',[1 size(im2,1)]);
    
    % correct for image vs. cartesian coords by flipping image along vertical axis
    im2_trans = flipud(im2_trans);

    % imwarp crashes
    %{
    imref_R = imref2d(size(im2));
    tform = affine2d(A);
    im2_trans = imwarp(im2,tform,'nearest','OutputView',imref_R);
    %}
    
    % test
    [xp_test] = tformfwd(tform, x);

    bfsave(im2_trans, strcat('reg_',fname_list{i}));
    
    clear x
end



% test out built in similarity transform
%tform = fitgeotrans(x,xp,'Similarity');
    
% visualize
%{
figure,
imagesc(im1),
figure,
imagesc(im2),
figure,
imagesc(im2_trans)
axis([1 1024 1 1024])
%}