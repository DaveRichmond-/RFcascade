function [sp] = LK_CompWarps(s0,Sj,Sja,weights,q,p,dq,dp);

% compose two warps, each of which is a composition => N(W(x;p);q) . N(W(x;-dp);-dq)
% sp : destination of s0 under the composition of the two warps


% first warp: N(W(s0; -dp); -dq)
[st1, s1] = LK_Warp(s0,Sj,Sja,weights,-dq,-dp);

% second warp: N(W(s0; p); q)
[st2, s2] = LK_Warp(s0,Sj,Sja,weights,q,p);     % NOTE: THIS WARP IS COMPUTED IN THE FIRST STEP OF THE LK ITERATION.  JUST PASS IT FORWARD FROM THERE.

% combine these two warps to find the destination of the base-mesh vertices under the composition of the two warps
sp = fnval(st2, s1);