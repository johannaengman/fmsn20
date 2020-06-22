function negloglike = gmrf_negloglike_NG(theta, y, A, B, G, E, error_term, isCAR)
% GMRF_NEGLOGLIKE_NG_SKELETON Calculate the GMRF likelihood for non-Gaussian observations
%
% negloglike = GMRF_negloglike_NG(theta, y, A, B, G, E)
%
% theta - log of parameters
% y - data vector, as a column with n elements
% A - Observation matrix, sparse n-by-N
% B - Matrix of covariates, matrix of size n-by-Nbeta 
% C,G,G2 = matrices used to build a Matern-like CAR/SAR precision,
%          see matern_prec_matrices, sparse N-by-N
% qbeta = Precision for the regression parameters, scalar.
% E = The population count in each region, used for Poisson observations
%

% $Id: gmrf_negloglike_NG_skeleton.m 5109 2017-11-12 20:08:26Z johanl $

%ensure that E=1 if E not given (i.e. same/no population weight in all regions)
%if nargin<9, E=1; end

%extract parameters (and transform to natural parameters)
tau = exp(theta(1));
q_e = exp(theta(2));

n = size(G,1);

%compute Q matrices  (for an intrinsic CAR(1) or SAR(1) process)
if isCAR
    Q_x = tau*G;
else
    Q_x = tau*G*G;
end
%compute Q for beta-parameters
Qbeta = 1e-6*speye(size(B,2));

%combine all components of Q using blkdiag
%also compute the observation matrix by combining A and B matrices

if error_term
    Qall = blkdiag(Q_x, q_e*speye(size(A,2)), Qbeta);
    Aall = [A A B];
else
    Qall = blkdiag(Q_x, Qbeta);
    Aall = [A B];
end

%declare x_mode as global so that we start subsequent optimisations from
%the previous mode (speeds up nested optimisation).
global x_mode;
if isempty(x_mode)
  %no existing mode, use zeros as start
  x_mode = zeros(size(Qall,1),1);
end

%compute reorder (if needed)
p = amd(Qall + Aall'*Aall); %sparsity and reorder of Q_xy
Qall = Qall(p,p); %reorder
Aall = Aall(:,p);
x_mode = x_mode(p);

%find mode - using Newton-Raphson optimisation
x_mode = fminNR(@(x) gmrf_taylor(x, y, Aall, Qall, E), x_mode);

%find the Laplace approximation of the denominator co8mputed at the mode
[logp, ~, Q_xy] = gmrf_taylor(x_mode, y, Aall, Qall, E);
%note that logp = -log_obs + x_mode'*Q*x_mode/2.

%Compute choleskey factor of Q_xy
[R_xy, p_xy] = chol(Q_xy);
if p_xy~=0
  %choleskey factor fail -> (almost) semidefinite matrix -> 
  %-> det(Q) ~ 0 -> log(det(Q)) ~ -inf -> negloglike ~ inf
  %Set negloglike to a REALLY big value
  negloglike = realmax;
  return;
end
if error_term
    log_det_Q = n*log(tau) + n*log(q_e);
else
    log_det_Q = n*log(tau);
end

log_det_Q_xy = sum(log(diag(R_xy)));
%also compute determinant of Qall (might be simplified)

%note that logp = -log_obs + x_mode'*Q*x_mode/2.
negloglike = (logp - 0.5*log_det_Q) + log_det_Q_xy;

%inverse reorder before returning
x_mode(p) = x_mode;

%print diagnostic information (progress)
fprintf(1, 'fval: %11.4e%11.4e%11.4e\n', negloglike, theta(1), theta(2));
