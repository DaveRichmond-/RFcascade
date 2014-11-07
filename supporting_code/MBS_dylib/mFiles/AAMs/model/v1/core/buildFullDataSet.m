function [dataSet, fname_list] = buildFullDataSet(data_dir, somite_nums)

% builds data set from series of .txt files containing somite landmarks
% structure of the data table
% column 1 : point #
% column 2 : x
% column 3 : y
% column 4 : slice #
% column 5 : color #
% column 6 : ID #

% initialization

dataSet = [];

% get list of text files, store in fname_list (cell)

fname_list = getFileNames(data_dir, '.txt');

%{
listing = dir(data_dir);
cd(data_dir);

k=1;

for i = 1:length(listing),
    
    if length(strfind(listing(i).name,'.txt') >= 0),

        fname_list{k} = listing(i).name;
        k = k+1;
        
    end
    
end
clear i k listing
%}

for i = 1:length(fname_list),
    
    [headers, table] = readTextFiles2(fname_list{i}, 1, 6);
    table(:,7) = repmat(somite_nums(i), [size(table,1),1]);
    
    dataSet(size(dataSet,1)+1 : size(dataSet,1)+size(table,1),:) = table;
    
end

% correct for zero-indexing of Fiji

dataSet(:,[2:3]) = dataSet(:,[2:3]) + ones(size(dataSet,1),2);