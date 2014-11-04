%% 

%%
clear all,

% USER PARAMETERS %%%%%%%%%%%%%%%%%%%

num_synth = 5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

base = '/Users/richmond/Data/Somites/Processed/all+synthetic';

fname_list_gray = getFileNames(strcat(base, '/original/grayscale'), '.tif');
fname_list_feat = getFileNames(strcat(base, '/original/features'), '.tif');
fname_list_label = getFileNames(strcat(base, '/original/labels'), '.tif');


for i = 1:length(fname_list_gray),
    
    cd(strcat(base, '/original/grayscale'));
    grayscale = imread(fname_list_gray{i});

    cd(strcat(base, '/original/feat'));
    features = bf_openStack(fname_list_feat{i});

    cd(strcat(base, '/original/label'));
    label = imread(fname_list_label{i});
    
    rnd_rot = rand(1,num_synth)*20 - 10;
    
    for j = 1:num_synth,
        
        grayscale_rot = ;
        
        % save
        imwrite(int8(label_registered(:,:,i)), strcat(base,'/synthetic/labels/',fname_list_label,'','.tif'))
        
        bfsave(scaleImage(grayscale_registered(:,:,i), '16bit', 1), strcat('grayscale_registered',num2str(offset_count + i),'.tif'))
        
    end
    
end

