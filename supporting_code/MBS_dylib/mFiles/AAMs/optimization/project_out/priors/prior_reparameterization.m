function [pDerivs] = prior_reparameterization(s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,q,p,dq,dp,qp,pp);

% important: for computational efficiency, inherit the variables in the first 25 lines (or so) from the current state of the LK program

% calculate position s1
s1 = s0;

for i = 1:size(Sj,3),    
    s1 = s1 - dp(i)*Sj(:,:,i);
end

% calculate position s2
[st2, s2] = LK_Warp(s0,Sj,Sja,weights,-dq,-dp);

% calculate parameters of TSP Warp
[st3, s_temp] = LK_Warp(s0,Sj,Sja,weights,q,p);

% combine these two warps to find the destination of the base-mesh vertices under the composition of the two warps
Sc = fnval(st3, s2);

% parse coefficients of TSP for simplicity 
a = st3.coefs(:,9:11);
W = st3.coefs(:,1:8);

% TOP LEFT OF BLOCK MATRIX ------------>

% original way
%{
for k=1:8,
    for j=1:8,
        for i=1:2,
            
            WdUdDq1(i,j,k) = 2 * W(i,k) * [s1(1,j)*(s0(1,k)-s2(1,j)) + s1(2,j)*(s0(2,k)-s2(2,j))] * ...
                [1 + log( (s0(1,k)-s2(1,j))^2 + (s0(2,k)-s2(2,j))^2 )];

        end
    end
end



dScdDq1 = (1/weights(1))*( -a(:,1:2)*s1 + sum(WdUdDq1,3) );
%dqpdDq1 = Sja_vec' * reshape((dScdDq1 - s0)',[prod(size(s0)),1]);
%}

% re-usable terms
for k=1:8,
    for j=1:8,
        scnd_term = 1 + log((s0(1,k) - s2(1,j))^2 + (s0(2,k) - s2(2,j))^2);
        dUdx2(k,j) = -2 * (s0(1,k) - s2(1,j)) * scnd_term;
        dUdy2(k,j) = -2 * (s0(2,k) - s2(2,j)) * scnd_term;
    end
end

% calc dScdx2 and dScdy2 -------------->
dScdx2 = a(:,1)*ones(1,size(s0,2)) + W*dUdx2;
dScdy2 = a(:,2)*ones(1,size(s0,2)) + W*dUdy2;

% calc dx2dDqi and dy2dDqi ------------>
dx2dDq(:,:,1) = (-1/weights(1))*ones(size(s0,1),1)*s1(1,:);
dy2dDq(:,:,1) = (-1/weights(1))*ones(size(s0,1),1)*s1(2,:);

dx2dDq(:,:,2) = ( 1/weights(2))*ones(size(s0,1),1)*s1(2,:);
dy2dDq(:,:,2) = (-1/weights(2))*ones(size(s0,1),1)*s1(1,:);

dx2dDq(:,:,3) = (-1/weights(3))*ones(size(s0));
dy2dDq(:,:,3) = zeros(size(s0));

dx2dDq(:,:,4) = zeros(size(s0));
dy2dDq(:,:,4) = (-1/weights(4))*ones(size(s0));;

% calc dScdDqi
for i = 1:4,
    dScdDq(:,:,i) = dScdx2.*dx2dDq(:,:,i) + dScdy2.*dy2dDq(:,:,i);
end

dScdDq_vec = squeeze(reshape(permute(dScdDq, [2 1 3]), [prod(size(s0)) 1 4]));

dqpdDq = Sja_vec' * dScdDq_vec;

% TOP RIGHT OF BLOCK MATRIX

% dx2dDpi
for i=1:length(dp),
    
    dx2dDp(:,:,i) = -(1-dq(1)/weights(1))*ones(size(s0,1),1)*Sj(1,:,i) - ...
        (dq(2)/weights(2))*ones(size(s0,1),1)*Sj(2,:,i);

    dy2dDp(:,:,i) = (dq(2)/weights(2))*ones(size(s0,1),1)*Sj(1,:,i) - ...
        (1-dq(1)/weights(1))*ones(size(s0,1),1)*Sj(2,:,i);

    dScdDp(:,:,i) = dScdx2.*dx2dDp(:,:,i) + dScdy2.*dy2dDp(:,:,i);
    
end
        
dScdDp_vec = squeeze(reshape(permute(dScdDp, [2 1 3]), [prod(size(s0)) 1 length(dp)]));
dqpdDp = Sja_vec' * dScdDp_vec;

% calc dqpdDqp
%{
dScdDqp_vec = [dScdDq_vec, dScdDp_vec];
dqpdDqp = Sja_vec' * (dScdDqp_vec - repmat(s0_vec,[1,4+length(dp)]));
%}

% BOTTOM LEFT OF BLOCK MATRIX ------------>

[A,b] = LK_qtoA(qp, weights);
invA = A^-1;

for j = 1:4,
    
    dAdDq(:,:,j) = (1/weights(1))*eye(size(s0,1))*dqpdDq(1,j) + (1/weights(2))*[0 -1; 1 0]*dqpdDq(2,j);
    
    dbdDq(:,:,j) = (1/weights(3))*[1;0]*ones(1,size(s0,2))*dqpdDq(3,j) + (1/weights(4))*[0;1]*ones(1,size(s0,2))*dqpdDq(4,j);
    
    dNinvdDq(:,:,j) = (-invA*dAdDq(:,:,j)*invA)*(Sc - repmat(b,[1, size(s0,2)])) + (invA * (dScdDq(:,:,j) - dbdDq(:,:,j)));
    
end

dNinvdDq_vec = squeeze(reshape(permute(dNinvdDq, [2 1 3]), [prod(size(s0)) 1 4]));

dppdDq = Sj_vec' * dNinvdDq_vec;

% BOTTOM RIGHT OF BLOCK MATRIX ------------>

for j = 1:length(dp),
    
    dAdDp(:,:,j) = (1/weights(1))*eye(size(s0,1))*dqpdDp(1,j) + (1/weights(2))*[0 -1; 1 0]*dqpdDp(2,j);
    
    dbdDp(:,:,j) = (1/weights(3))*[1;0]*ones(1,size(s0,2))*dqpdDp(3,j) + (1/weights(4))*[0;1]*ones(1,size(s0,2))*dqpdDp(4,j);
    
    dNinvdDp(:,:,j) = (-invA*dAdDp(:,:,j)*invA)*(Sc - repmat(b,[1, size(s0,2)])) + (invA * (dScdDp(:,:,j) - dbdDp(:,:,j)));
    
end

dNinvdDp_vec = squeeze(reshape(permute(dNinvdDp, [2 1 3]), [prod(size(s0)) 1 length(dp)]));

dppdDp = Sj_vec' * dNinvdDp_vec;

% reconstruct full matrix of partial derivatives
pDerivs = [dqpdDq, dqpdDp; dppdDq, dppdDp];