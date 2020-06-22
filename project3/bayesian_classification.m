function [z, N, n_correct, Y] = bayesian_classification(theta, x, im_col)

sigma2_eps = theta(1);
pc = theta(4);

pN = (1/(sqrt(2*pi*sigma2_eps)))*exp(-(im_col-x).^2/(2*sigma2_eps));
p_z = (pN*pc)./(pN*pc+(1-pc));
rand_z = rand(length(im_col), 1);

z = (rand_z < p_z);
n_correct = sum(z);
N = length(z);

Y = im_col(z);
end