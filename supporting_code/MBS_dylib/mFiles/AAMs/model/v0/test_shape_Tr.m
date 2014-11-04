%% description

% this script runs through the protocol for generating a shape model from data points 
% according to Cootes and Taylor, Stat Models of Appearance for CV

% structure of the data table
% column 1 : point #
% column 2 : x
% column 3 : y
% column 4 : slice #
% column 5 : color #
% column 6 : ID #

%% user inputs (for now)

fname = 'raw.txt';      %  
arraySize = [(51-23+1)*8,6];                    %
numHeaderObjs = 6;                      %
d = 2;                                  % dimensionality of landmarks (ie, images)

%% load data

[headers, table] = readTextFiles(fname, numHeaderObjs, arraySize);

%% put window of good data into training set

% trainingSet = [];
window = [1, arraySize(1)/8];
windowlength = window(2)-window(1)+1;

trainingSet(size(trainingSet,1)+1 : size(trainingSet,1)+windowlength*8, :) = ...
    table((window(1)-1)*8 + 1 : (window(1)-1)*8 + windowlength*8, :);

%% reformat data for ease of manipulation

points = unique(table(:,1));
numpoints = 8;                                  % b/c there was a labeling screw-up for somite 20!
%numpoints = points(end) - points(1) + 1;

numslices = size(table,1)/numpoints;            % b/c the frame # shows up repeatedly
%slices = unique(table(:,4));
%numslices = slices(end) - slices(1) + 1;

% if (numslices*numpoints ~= arraySize(1)),
%     error('data not parsed correctly'),
% end

xmat = zeros(numpoints, d, numslices);
for i=1:numslices,
    xmat(:,:,i) = table((i-1)*numpoints + 1 : i*numpoints, 2:3);
end

%% center all shapes on the origin

for i = 1:numslices,
    
    xmat(:,:,i) = xmat(:,:,i) - repmat(mean(xmat(:,:,i),1),[numpoints 1]);
    
end

%% plot all shapes as overlay to see if the time window is good

figure, hold on,

for i = 1:numslices,
    
    plot(xmat(:,1,i),xmat(:,2,i),'ko'),
    
end

print -dtiff unnormalized_points.tif

%% transfer to vector form (mix x and y pairs into single column)

xvec = zeros(numpoints * d, numslices);

for i=1:numslices,
    
    xvec(:,i) = [xmat(:,1,i); xmat(:,2,i)];
    
end

%% pick one example to be mean shape, and scale all vectors by the norm of this mean

meanslice = 105;
%meanslice = ceil(numslices/2);

xvec = xvec / norm(xvec(:,meanslice));
xbar_0 = xvec(:,meanslice);

%% iterative alignment of all vectors with mean, to define mean

thresh = 1e-6;


D = 2*thresh;
xbar_curr = xbar_0;
k = 1;

while D(k) > thresh,
    
    k = k+1;
    
    % align all vectors to current estimate of mean
    
    for j=1:numslices,
        
        xvec(:,j) = alignvectors(xvec(:,j), xbar_curr);
        
        % normalize xvec according to tangent method
    
        xvec(:,j) = xvec(:,j) / dot(xvec(:,j),xbar_curr);
        
    end
    
    % estimate new mean from xvec
    
    xbar_new = mean(xvec,2);
    
    % align and normalize new estimate of mean
    
    % xbar_new = alignvectors(xbar_new, xbar_0);
    xbar_new = xbar_new / norm(xbar_new);
    
    % calculate difference, for recursion limit
    
    D(k) = norm(xbar_new - xbar_curr);
    
    % re-assign new estimate to current estimate for next loop
    
    xbar_curr = xbar_new;
    
end

%% plot xvec to test above routine

figure,
hold on,

for i=1:size(xvec,2),
    
    plot(xvec(1:numpoints,i), xvec(numpoints+1:end,i),'ko')
    
end

print -dtiff normalized_points.tif
 
%% PCA

xbar = mean(xvec, 2);
s = numslices;
n = numpoints*2;            % should be = size(xvec,1)

R = xvec - repmat(xbar, [1 s]);
Sigma = (R*R')/(s-1);

[U, S, V] = svd(Sigma);

%{
% compress onto minimum # dimensions that retains > 95% of total variance
f = 0.95;

k = 0;
while var_lost < (1-f),
    
    k = k+1;
    
    Ureduce = U(:,1:k);
    Z = Ureduce' * R;
    
    avg_error = sum(sum((R-Z).^2,1),2)/s;
    total_variation = sum(sum((R).^2,1),2)/s;
    
    var_lost = avg_error / total_variation;
    
end
%}

for k=1:size(U,2),
        
    Ureduce = U(:,1:k);
    Z = Ureduce' * R;
    X_approx = Ureduce * Z;
    
    avg_error = sum(sum((R - X_approx).^2,1),2)/s;
    total_variation = sum(sum((R).^2,1),2)/s;
    
    var_lost(k) = avg_error / total_variation;
    
end

figure,
plot(var_lost,'r-','LineWidth',2),
xl = xlabel('number of PCA modes kept');
yl = ylabel('fraction of total variation LOST');
tit = title('Performance of PCA decomposition');
set(xl,'fontsize',20),
set(yl,'fontsize',20),
set(tit,'fontsize',24),

print -dtiff PCA_varLost_#modes.tif

%% explore shape variation along principle axes

% user parameters to make a movie of the shape variation

p_axis = 1;
createMovie = 1;

%

if createMovie,

    fname = strcat('variation along principle axis ',num2str(p_axis),'.mp4');
    writerObj = VideoWriter(fname,'MPEG-4');
    open(writerObj);
    
end

Psi = U;
lambda = S(p_axis);
sigma_p = sqrt(lambda);

deltaS = 6*sigma_p/99;
range = [ [0 : deltaS : 3*sigma_p], [3*sigma_p : -deltaS : -3*sigma_p], [-3*sigma_p : deltaS : 0] ];

fhandle = figure;

%{
F = getframe(fhandle);
image(F.cdata)
colormap(F.colormap)
%}

for i=1:length(range);
    
    x_1 = xbar + Psi(:,p_axis)*range(i);
    plot(x_1(1:numpoints), x_1(numpoints+1:2*numpoints),'o',...
        'LineWidth',2,...
        'MarkerEdgeColor','k',...
        'MarkerSize',10),
    hold on
    fnplt(cscvn([[x_1(1:numpoints);x_1(1)]';[x_1(numpoints+1:2*numpoints);x_1(numpoints+1)]']),...
        'r',2)
    hold off
    
    axis([-0.6 0.6 -0.6 0.6]),
    set(gca,'XTick',[],'YTick',[]),
    
    if createMovie,
        
        frame = getframe(fhandle);
        writeVideo(writerObj,frame);
        
    else
        
        pause(0.5),
    
    end
    
end

if createMovie,
    
    close(writerObj),
    
end


%% track PCA weights through time

% user parameters for plotting

b_element = 1;

% 

b = Psi'*R;

h = figure,
plot(b(b_element,:),'LineWidth',2),

xl = xlabel('Time [frame #]'); set(xl,'fontsize',20),
yl = ylabel(strcat('Weight b_',num2str(b_element))); set(yl,'fontsize',20),
tit = title('Description of somite shape through time'); set(tit,'fontsize',24),

%print(h,'-dtiff',strcat('b',num2str(b_element),'_time.tif')),

%% plot primary 2 modes

h = figure, hold on,
plot(b(2,:),b(1,:),'LineWidth',2)
plot(b(2,:),b(1,:),'ro','LineWidth',2)

xl = xlabel('Weight b_2'); set(xl,'fontsize',20),
yl = ylabel('Weight b_1'); set(yl,'fontsize',20),
tit = title('Trajectory of two primary PCA weights'); set(tit,'fontsize',24),

print(h,'-dtiff',strcat('b2_b1.tif')),

%% re-assign b-values to individual trajectories

som08_b1 = b(1, 1:25);
som09_b1 = b(1, 26:55);
som10_b1 = b(1, 56:91);
som11_b1 = b(1, 92:126);
som12_b1 = b(1, 127:149);
som13_b1 = b(1, 150:179);
som14_b1 = b(1, 180:208);

h = figure, hold on,
plot(7+[1:25]-1,  som08_b1, '-k','LineWidth',2),
plot(10+[1:30]-1,  som09_b1, '-r','LineWidth',2),
plot([16:51],  som10_b1, '-b','LineWidth',2),
plot([17:51],  som11_b1, '-g','LineWidth',2),
plot([23:45], som12_b1, '-c','LineWidth',2),
plot([22:51], som13_b1, '-m','LineWidth',2),
plot([23:51], som14_b1, '-k','LineWidth',2),

leg = legend('Somite #8','Somite #9','Somite #10','Somite #11','Somite #12','Somite #13','Somite #14'),
set(leg,'fontsize',16),

%% movie from single trajectory

fname = strcat('Fit som18 with model derived from training set.mp4');
writerObj = VideoWriter(fname,'MPEG-4');
open(writerObj);

fhandle = figure;

for i=1:length(som11_b1);
    
    x_1 = xbar + Psi(:,1)*som11_b1(i);
    plot(x_1(1:numpoints), x_1(numpoints+1:2*numpoints),'o',...
        'LineWidth',2,...
        'MarkerEdgeColor','k',...
        'MarkerSize',10),
    hold on
    fnplt(cscvn([[x_1(1:numpoints);x_1(1)]';[x_1(numpoints+1:2*numpoints);x_1(numpoints+1)]']),...
        'r',2)
    hold off
    
    axis([-0.6 0.6 -0.6 0.6]),
    set(gca,'XTick',[],'YTick',[]),
    
    frame = getframe(fhandle);
    writeVideo(writerObj,frame);
        
end

close(writerObj),