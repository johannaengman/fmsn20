function [sigma2_eps, tau] =  gamma_invgamma(theta, x, u, z, y, N, n_correct, Q_x)

% y = Y(observed)
% N number of pixels
% n_correct - number of assumed correct pixels

sigma2_eps = theta(1);
kappa = theta(2);
tau = theta(3);
pc = theta(4);

a = n_correct/2+1;
b = 2/sum((x(z)-y)'*(x(z)-y))
sigma2_inv = gamrnd(a, b);
sigma2_eps = 1./sigma2_inv;
   
a = N/2 + 1;
b = 2/(u'*Q_x*u);
tau = gamrnd(a, b);

end