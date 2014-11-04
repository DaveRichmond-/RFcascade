function [trainingSet, fname_list, windows_record] = buildTrainingSet(data_dir, somite_nums);

% initialization

trainingSet = [];

% get list of text files, store in fname (cell)

listing = dir(data_dir);

k=1;

for i = 1:length(listing),
    
    if length(strfind(listing(i).name,'.txt') >= 0),

        fname_list{k} = listing(i).name;
        k = k+1;
        
    end
    
end
clear i k listing

% do PCA one somite at a time, to determine appropriate window

for i = 1:length(fname_list),
    
    [headers, table] = readTextFiles2(fname_list{i}, 1, 6);
    table(:,7) = ones(size(table,1), 1) * somite_nums(i);
    
    [R, Sigma, Psi, Lambda, V] = make_shape_model(table, 2);
    
    [b_fit] = fit_shape_model(Psi, R, 1);
    
    % prompt user for window to use in training set
    
    frames = unique(table(:, 4));
    plot(frames, b_fit, 'Linewidth', 2),
    axis('tight'),
    
    disp(strcat('first frame = ', num2str(frames(1)))),
    first_frame = input('input first frame of window: ');
    disp(strcat('last frame = ', num2str(frames(end)))),
    last_frame = input('input last frame of window: ');
    window = [first_frame : last_frame];
    
    % copy respective frames to trainingSet
    
    for j = 1:length(window),
        
        curr_index = find(table(:, 4) == window(j));
        trainingSet(size(trainingSet,1)+1 : size(trainingSet,1)+length(curr_index), :) = table(curr_index, :);
        
    end
    
    % store window
    windows_record(i, :) = [somite_nums(i) window(1) window(end)];
    
end

% correct for zero-indexing of Fiji

trainingSet(:,[2:3]) = trainingSet(:,[2:3]) + ones(size(trainingSet,1),2);