function [x_samp, u_samp] = reconstruction_GMRF(theta, Q_x, Y, z, im)

% theta = [sigma2_eps, kappa, tau, pc];
% Y - data vector, as a column with n elements
% A - Observation matrix, sparse n-by-N
% B - Matrix of covariates, matrix of size n-by-Nbeta 
% C,G,G2 = matrices used to build a Matern-like CAR/SAR precision,
%          see matern_prec_matrices, sparse N-by-N

sigma2_eps = theta(1);
tau = theta(3);

A = sparse(1:sum(z), find(z), 1, sum(z), size(Q_x,1));
A_tilde = [A ones(length(Y),1)];

Q_x = tau*Q_x;

Qbeta = 1e-6 * speye(1); 
AtA = (A_tilde'*A_tilde)./sigma2_eps;
Qall = blkdiag(Q_x, Qbeta);
Q_xy = Qall + AtA;

p = amd(Q_xy); %sparsity and reorder of Q_xy
Q_xy = Q_xy(p,p); %reorder
A_tilde = A_tilde(:,p);

R_xy = chol(Q_xy);
E_xy = R_xy\((A_tilde'*Y/sigma2_eps)'/R_xy)';


e = randn(size(R_xy,1),1);
u_samp = E_xy + R_xy\e;
u_samp(p) = u_samp;
A_tilde = [speye(numel(im)) ones(numel(im),1)];

x_samp = A_tilde*u_samp;
u_samp = u_samp(1:end-1);
end