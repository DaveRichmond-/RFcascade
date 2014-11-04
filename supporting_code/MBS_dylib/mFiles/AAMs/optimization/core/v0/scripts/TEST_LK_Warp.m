%% script to test applying warp

% set parameters of warp arbitrarily (these will be input from last iteration of LK)

q = [0 0 5 -2];     % coefficients of similarity transformation
p = shape(1).b(:,10)';  % coefficients of shape for 10th frame of somite 7

p = zeros(size(p));

% run LK_Warp

st = LK_Warp(s0,Sj,Sja,weights,q.*weights',p);
st.coefs,