%% build AR model for somite evolution in time

%load('fit_to_LOO_model_s7-15.mat');

clear all

load('GT_priors.mat');

% discard data from som7, which i will use to test later
params = params(2:end);
num_real = size(params,2);

%
for k=1:num_real,
    % assign first 59 time points (so that all data has the same length)
    % v(:,:,k) = [params(k).q(1:4,1:59); params(k).p(1,1:59); params(k).L(1,1:59)]';
    v(:,:,k) = [params(k).p(1,1:59)]';
end

v = flipdim(v,1);

%%

pmin = 2;
pmax = 2;

[w,A,C,SBC,FPE,th] = arfit(v,pmin,pmax);

%% test model

[siglev,res]=arres(w,A,v);

%% simulate
%{
for j = 1:1e2,
    v_sim(:,:,j) = arsim(w,A,C,58);
end

figure,
for i = 1:size(v,2),
    
    subplot(1,size(v,2),i), 
    hold on,
    
    for j = 1:size(v_sim,3),
        plot(v_sim(:,i,j),'r'),
    end
    for j = 1:size(v,3),
        plot(v(:,i,j)),
    end
    
end
%}
%% make predictions based on AR model

track_v = squeeze(v(:,1,1));

v_pred = track_v(1:10);

for i = 11:length(track_v),
    
    v_past = [v_pred(i-1); v_pred(i-2)];
    v_pred(i) = A*v_past + w;
    
end

%%

figure,
plot(track_v)
hold on,
plot(v_pred,'r')
plot([0; track_v(1:end-1)],'k')
leg = legend('data','pred','shifted data');
set(leg,'fontsize',14)
