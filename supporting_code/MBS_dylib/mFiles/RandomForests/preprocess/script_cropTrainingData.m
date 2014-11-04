%% crop training data for RF

%% set variables

% copy topleftPos from .Numbers into workspace

topleftPos = repmat(topleftPos, [3, 1]);

%%

width = 800;
height = 900;

imageDir = '/Users/richmond/Data/Somites/Processed/First set/registered/Features/without_XY/Train';

fname_list = getFileNames(imageDir, 'tif');

cd(imageDir);
mkdir('crop');

for i = 1:length(fname_list)
    
    % for 2d images
    %{
    im = imread(fname_list{i});
    im = im(topleftPos(i,2):topleftPos(i,2)+height-1, topleftPos(i,1):topleftPos(i,1)+width-1);
    %}

    % for stacks
    [im,sizeC,sizeZ,sizeT] = bf_openStack(fname_list{i});
    im = im(topleftPos(i,2):topleftPos(i,2)+height-1, topleftPos(i,1):topleftPos(i,1)+width-1, :);
    
    output_name = strcat('crop/',fname_list{i});
    bfsave(im, output_name);
    
end