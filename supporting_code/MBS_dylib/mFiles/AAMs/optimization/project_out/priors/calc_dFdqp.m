function [F, dFdqp] = calc_dFdqp(r1,rc,c,q,p,dq,dp,weights,s0,Sj,Sja,pos_num);

%

%(2) distance between position r1 and r2 
dist = r1 - rc;
dist = dist(:);

%(3) calc F
F = c*norm(dist);

%(4) matrix A of affine transform under q (i.e. the 2nd Warp in the composition)
[A,b] = LK_qtoA(q, weights);

%(6) calc position s2 after warp W(s0; -dq, -dp)
[st, s2] = LK_Warp(s0,Sj,Sja,weights,-dq,-dp);

x2 = s2(1,pos_num);
y2 = s2(2,pos_num);

Phi_x2y2 = stcol(s0, [x2; y2], 'tr');

%(8) calc equivalent of L^-1
[d ns0] = size(s0);
P = [s0.', ones(ns0,1)];

% QR decomposition of P
[Q,R] = qr(P);
Q1 = Q(:, 1:d+1);
Q(:, 1:d+1) = [];

% calculate colocation matrix, K
K = stcol(s0, s0, 'tr');

% calculate A,C which act like L^-1, decomposed into affine and non-affine transformations
TPS_na = (Q / (Q'*K*Q)) * Q';                                    % coefficients of non-affine trans
TPS_a = ((eye(size(C*K)) - TPS_na*K)*Q1) / (R(1:d+1, 1:d+1).');      % coefficients of affine trans

%(9) calc M = L^-1 * [Phi; x2; y2; 1]
M = [TPS_na', TPS_a] * [Phi_x2y2; x2; y2; 1];

% 
[st, s1] = LK_Warp(s0,Sj,Sja,weights,zeros(4,1),p);

dFdq(1) = ((c^2)/F)*dist'*( (1/w1)*eye(2)*s1 * M);
dFdq(2) = ((c^2)/F)*dist'*( (1/w2)*[0 -1; 1 0]*s1 * M);
dFdq(3) = ((c^2)/F)*dist'*( (1/w3)*[1;0]*ones(1,size(s0,2)) * M);
dFdq(4) = ((c^2)/F)*dist'*( (1/w4)*[0;1]*ones(1,size(s0,2)) * M);

for j=1:length(p),
    
    dFdp(j) = ((c^2)/F)*dist'*( A*Sj(:,:,j) * M );
    
end

dFdqp = [dFdq; dFdp];