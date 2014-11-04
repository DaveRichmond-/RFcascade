function [F, dFdqp] = calc_dFdqp_POS(r1,rc,c,q,p,weights,s0,Sj,Sja,pos_num);

%(1) distance between position r1 and rc 
dist = r1 - rc;
dist = dist(:);

%(2) calc F
F = c*norm(dist);

%(3) matrix A of affine transform under q (i.e. the 2nd Warp in the composition)
[A,b] = LK_qtoA(q, weights);

%(4) calc position s1 after warp W(s0; 0, p)
[st, s1] = LK_Warp(s0,Sj,Sja,weights,zeros(4,1),p);

%(5) dFdq
dFdq(1) = ((c^2)/F)*dist'*( (1/weights(1))*eye(2)*s1(:,pos_num) );
dFdq(2) = ((c^2)/F)*dist'*( (1/weights(2))*[0 -1; 1 0]*s1(:,pos_num) );
dFdq(3) = ((c^2)/F)*dist'*( (1/weights(3))*[1;0] );
dFdq(4) = ((c^2)/F)*dist'*( (1/weights(4))*[0;1] );

%(5) dFdp
for j=1:length(p),
    dFdp(j) = ((c^2)/F)*dist'*( A*Sj(:,pos_num,j) );
end

% concatenate
dFdqp = [dFdq(:); dFdp(:)];