%% load complete training set

clear all
load trainingSet.mat

%% make model from training set

[xbar, R, Sigma, Psi, Lambda, V] = make_shape_model(trainingSet, 2);

%% fit somite data to model based on trainingSet

% specify data set to fit

fname = 'somite08.txt';
[headers, table] = readTextFiles2(fname, 1, 6);

frames = unique(table(:,4));

for i=1:length(frames),
    
    % subtract mean from image
    indx = find(table(:,4) == frames(i));
    xmat = table(indx,2:3);
    xmat = xmat - repmat(mean(xmat,1),[size(xmat,1) 1]);
    x = xmat(:);
    
    % align image onto xbar
    
    [x_aligned] = alignvectors(x, xbar);
    
    % fit
    
    [b(i)] = fit_shape_model(Psi, x_aligned(:), 1);             % principle axis to fit along is set to 1

end


%% user parameters for plotting

h = figure,
plot(b,'LineWidth',2),

xl = xlabel('Time [frame #]'); set(xl,'fontsize',20),
yl = ylabel(strcat('Weight b_',num2str(1))); set(yl,'fontsize',20),
tit = title('Description of somite shape through time'); set(tit,'fontsize',24),

%print(h,'-dtiff',strcat('b',num2str(b_element),'_time.tif')),
