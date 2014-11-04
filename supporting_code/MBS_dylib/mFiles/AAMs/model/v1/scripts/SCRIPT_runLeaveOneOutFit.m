%% user
clear all

image_fname = 'image.tif';
data_fname = 'fullDataSet.mat';

%% load data

[imageStack,sizeC,sizeZ,sizeT] = bf_openStack(image_fname);
load(data_fname);

%% run analysis

SFPpoints = [0:7];

[shape, appear] = leaveOneOutFit(imageStack, dataSet, SFPpoints(:));

%% PLOTTING, ETC. ------------------------------>

%% shape and appearance data vs. time

s_axis = 1;
a_axis = 1;

plot_fit2AAM(shape, s_axis, appear, a_axis)

%% variance in primary modes

% shape

Lambda = diag(shape(1).Lambda);

var_ret = 1 - cumsum(Lambda)/sum(Lambda);

figure,
plot(var_ret,'r-','LineWidth',2),
xl = xlabel('number of PCA modes kept');
yl = ylabel('fraction of total variation LOST');
tit = title('Performance of PCA decomposition');
set(xl,'fontsize',20),
set(yl,'fontsize',20),
set(tit,'fontsize',24),

print -dtiff shape_var_lost.tif

%% create movie of shape model

s_axis = 1;
stdevs = 3;

shape_model_movie(shape(1).xbar_vector, shape(1).Psi, shape(1).Lambda, s_axis, stdevs);

%% plot instance of shape model

s_axis = 1;
b_val = 0;

shape_model_figure(shape(1).xbar_vector, shape(1).Psi, shape(1).Lambda, s_axis, b_val)

print('-dtiff',strcat('shape model_axis = ',num2str(s_axis),'_b = ',num2str(b_val),'.tif'))

%% plot fit of somite to shape model

som_num = 7;

shape_fit_movie(som_num, shape(1).xbar_vector, shape(1).Psi, shape(1).b(1,:), shape(1).frames)

%% create time vs. somite plot from shape data

%

bins = [0.1 : -0.05 : -0.15];
width = 0.025;
color_str = ['r','g','b','c','m','k'];

%

for i = 1:size(shape,2),
    
    %if ismember(shape(i).som_num, som_incl),
    
        som_nums(i) = shape(i).som_num;
        frames = shape(i).frames;
        b = shape(i).b(1,:);
        
        for j = 1:length(bins),
            
            
            indx = find( (b > (bins(j) - width)) .* (b < (bins(j) + width)) );
            t = frames(indx);
            
            med_t(j,i) = median(t);
            mean_t(j,i) = mean(t);
            
            %plot(som_num, med_t(j,i), strcat(color_str(mod(j-1,6)+1),'.'),'MarkerSize',20)
            
            clear indx t
            
        end
        
    %end
    
end

% replot for fitting

figure,
hold on

% plot med_t

for j = 1:length(bins),
    
    plot(som_nums, med_t(j,:), strcat(color_str(mod(j-1,6)+1),'.'),'MarkerSize',20),
    
end

% plot manually identified frame when somite is first visible

plot(manual_scores(:,1), manual_scores(:,2), '^', 'MarkerSize',10, 'MarkerEdgeColor','k','MarkerFaceColor','k') 

xl = xlabel('Somite #');
yl = ylabel('Frame #');
set(xl,'fontsize',20),
set(yl,'fontsize',20),
leg = legend('b(1) = 0.1','b(1) = 0.05','b(1) = 0.0','b(1) = -0.05','b(1) = -0.1','b(1) = -0.15','manual scores');
set(leg,'fontsize',12)
axis([(som_nums(1)-0.5) (som_nums(end)+0.5) 0 100])

% fit med_t

% choose b_vals for fitting

b_fits = [0.1, 0.0, -0.1];
fit_indx = zeros(1,length(b_fits));

%{
for i = 1:length(b_fits),
    
    fit_indx(i) = find(bins == b_fits(i));
    
end
%}
fit_indx = [1,3,5];

fitobject_man = fit(manual_scores(:,1), manual_scores(:,2), 'poly1');
fitobject_1 = fit(som_nums', med_t(fit_indx(1),:)', 'poly1');
fitobject_2 = fit(som_nums', med_t(fit_indx(2),:)', 'poly1');
fitobject_3 = fit(som_nums', med_t(fit_indx(3),:)', 'poly1');

% plot corresponding lines

line([0, 30],[fitobject_man.p2, (fitobject_man.p1*30 + fitobject_man.p2)], 'LineStyle', '--', 'Color', 'k')
line([0, 30],[fitobject_1.p2, (fitobject_1.p1*30 + fitobject_1.p2)], 'LineStyle', '--', 'Color', strcat(color_str(fit_indx(1))))
line([0, 30],[fitobject_2.p2, (fitobject_2.p1*30 + fitobject_2.p2)], 'LineStyle', '--', 'Color', strcat(color_str(fit_indx(2))))
line([0, 30],[fitobject_3.p2, (fitobject_3.p1*30 + fitobject_3.p2)], 'LineStyle', '--', 'Color', strcat(color_str(fit_indx(3))))

%% APPEARANCE --------------------------------->

%% variance in primary modes

% appearance

Lambda = appear(1).Lambda;

var_ret = 1 - cumsum(Lambda)/sum(Lambda);

figure,
plot(var_ret,'r-','LineWidth',2),
xl = xlabel('number of PCA modes kept');
yl = ylabel('fraction of total variation LOST');
tit = title('Performance of PCA decomposition');
set(xl,'fontsize',20),
set(yl,'fontsize',20),
set(tit,'fontsize',24),

print -dtiff appear_var_lost.tif

%% create movie of appearance model

a_axis = 1;
n_sigma = 3;

pixelList = cell2mat(appear(1).CC.PixelIdxList);

[mean_image, eigen_images] = mean_eigen_images(appear(1).gbar,appear(1).X,pixelList,appear(1).Psi);

appearance_model_movie(mean_image, eigen_images, appear(1).Lambda, a_axis, n_sigma);

%% plot instance of appearance model

a_axis = 1;
b_val = 0.0;

[mean_image, eigen_images] = mean_eigen_images(appear(1).gbar,appear(1).X,appear(1).CC,appear(1).Psi);
appearance_model_figure(mean_image, eigen_images, a_axis, b_val)

print('-dtiff',strcat('instance of appearance model_axis = ',num2str(a_axis),'_b = ',num2str(b_val),'.tif'))

%% fit data to model

som_num = 10;
[mean_image, eigen_images] = mean_eigen_images(appear(4).gbar,appear(4).X,appear(4).CC,appear(4).Psi);

appearance_fit_movie(som_num, mean_image, eigen_images, appear(4).b(1,:), appear(4).frames);

%% create time vs. somite plot from appearance data

%

bins = [-0.6 : 0.2 : 0.4];
width = 0.1;
color_str = ['r','g','b','c','m','k'];

%

for i = 1:size(appear,2),
    
    som_nums(i) = appear(i).som_num;
    frames = appear(i).frames;
    b = appear(i).b(1,:);
    
    for j = 1:length(bins),
        
        %{
        if j == 1,
            
            indx = find(b < (bins(j) + width));
                 
        elseif j == length(bins),
       
            indx = find(b > (bins(j) - width));
            
        else
        
            indx = find( (b > (bins(j) - width)) .* (b < (bins(j) + width)) );
        
        end
        %}
        indx = find( (b > (bins(j) - width)) .* (b < (bins(j) + width)) );
        t = frames(indx);
        
        med_t(j,i) = median(t);
        mean_t(j,i) = mean(t);
        
        clear indx t
        
    end
        
end


% replot for fitting

figure,
hold on

% plot med_t

for j = 1:length(bins),
    
    plot(som_nums, med_t(j,:), strcat(color_str(mod(j-1,6)+1),'.'),'MarkerSize',20),
    
end

% plot manually identified frame when somite is first visible

plot(manual_scores(:,1), manual_scores(:,2), '^', 'MarkerSize',10, 'MarkerEdgeColor','k','MarkerFaceColor','k') 

xl = xlabel('Somite #');
yl = ylabel('Frame #');
set(xl,'fontsize',20),
set(yl,'fontsize',20),
leg = legend('b(1) = -0.4','b(1) = -0.2','b(1) = 0.0','b(1) = 0.2','b(1) = 0.4');%,'b(1) = -0.1','b(1) = -0.15');%,'manual scores');
set(leg,'fontsize',12)
axis([(som_nums(1)-0.5) (som_nums(end)+0.5) 0 100])

% fit med_t

% choose b_vals for fitting

fit_indx = [1,3,5];

% fit med_t

fitobject_man = fit(manual_scores(:,1), manual_scores(:,2), 'poly1');
fitobject_1 = fit(som_nums', med_t(fit_indx(1),:)', 'poly1');
fitobject_2 = fit(som_nums', med_t(fit_indx(2),:)', 'poly1');
fitobject_3 = fit(som_nums(2:end)', med_t(fit_indx(3),2:end)', 'poly1');

% plot corresponding lines

line([0, 30],[fitobject_man.p2, (fitobject_man.p1*30 + fitobject_man.p2)], 'LineStyle','--', 'Color', 'k')
line([0, 30],[fitobject_1.p2, (fitobject_1.p1*30 + fitobject_1.p2)], 'LineStyle','--', 'Color', strcat(color_str(fit_indx(1))))
line([0, 30],[fitobject_2.p2, (fitobject_2.p1*30 + fitobject_2.p2)], 'LineStyle','--', 'Color', strcat(color_str(fit_indx(2))))
line([0, 30],[fitobject_3.p2, (fitobject_3.p1*30 + fitobject_3.p2)], 'LineStyle','--', 'Color', strcat(color_str(fit_indx(3))))
