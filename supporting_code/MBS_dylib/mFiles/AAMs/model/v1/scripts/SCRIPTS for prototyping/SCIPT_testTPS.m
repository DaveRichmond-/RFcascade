%% play around with thin plate splines calculations

% landmarks in initial image (x) and final image (y)

x = [0 -1 0 1; 1 0 -1 0];
y = [0 -1 0 1; 0.75 0.25 -1.25 0.25];

% useful params

[m nx] = size(x);
sizeval = size(y);
ny = sizeval(end);
sizeval(end) = [];
dy = prod(sizeval);


% 

colmat = stcol(x,x,'tr');

P = [x.', ones(nx, 1)];

[Q,R] = qr(P);

Q1 = Q(:,1:m+1),
Q(:,1:m+1) = [],

coefs1 = y * ((Q/(Q'*colmat*Q))*Q');
coefs2 = ((y - coefs1*colmat)*Q1)/(R(1:m+1,1:m+1).')

coefs = [coefs1, coefs2];