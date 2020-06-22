% Start with thetitan.jpgimage (or another grayscale image) and add 
% your own random noise (for titan.jpga correct implementation should 
% handle pc<0.5). 
clear
clc
close all
%%
tic
titan = imread('titan.jpg');
%house = rgb2gray(imread('house.jpg'));
%bar = rgb2gray(imread('bar.jpg'));
im = double(titan)/255;
im_org = double(titan)/255;

[m, n]= size(titan);
R = rand(m, n);
miss = 0.5; % p_c
known = (rand(m, n)>miss);

im(~known) = R(~known);
%%
figure
imagesc(im_org)
colormap(gray)

figure
imagesc(im)
colormap(gray)


%% Start, ilka värden ska vi gissa på?

sigma2_eps = var(im(:));
kappa = 0.1;
tau = 100;
pc = 0.5;

theta = [sigma2_eps, kappa, tau, pc];

%%

[u1,u2] = ndgrid(1:m,1:n);
[C,G,G2] = matern_prec_matrices([u1(:),u2(:)]);

Q_x = kappa^4*C + 2*kappa^2*G + G2;

im_col = im(:);
z = known(:);
Y = im_col(z);
n_correct = sum(z);
N = length(z);

%% Loopa över detta:
n_gibbs = 1000;
x_samp_tot = zeros(m, n, n_gibbs);
z_tot = zeros(m*n, n_gibbs);
tau_tot = zeros(1, n_gibbs);
sigma_tot = zeros(1, n_gibbs);
pc_tot = zeros(1, n_gibbs);


for i = 1:n_gibbs
    [x_samp, u_samp] = reconstruction_GMRF(theta, Q_x, Y, z, im);
    
    x_samp_tot(:,:,i) = reshape(x_samp, [m, n]);

    [z, N, n_correct, Y] = bayesian_classification(theta, x_samp, im_col);
    z_tot(:,i) = z;
    sum(z)
    [sigma2_eps, tau] = gamma_invgamma(theta, x_samp, u_samp, z, Y, N, n_correct, Q_x)
    
    if sigma2_eps == 0
        i
        sigma2_eps
        break;
    end
    sigma_tot(i) = sigma2_eps;
    tau_tot(i) = tau;
    
    theta(1) = sigma2_eps;
    theta(3) = tau;

    pc = beta_dist(n_correct, im_col)
    pc_tot(i) = pc;
    
    theta(4) = pc;
end
toc
%%

figure
subplot(2,2,1)
plot(tau_tot)
title('Tau')

subplot(222)
plot(sigma_tot)
title('Sigma')

subplot(223)
plot(pc_tot)
title('Pc')

subplot(224)
plot(sum(z_tot, 1)/(m*n))
title('z')

figure
imagesc(mean(x_samp_tot(:,:,end-99:end),3))
colormap(gray)

figure
imagesc(im_org-mean(x_samp_tot(:,:,end-99:end),3))
colormap(gray)

figure
imagesc(known - reshape(z, [m ,n]))
colormap(gray)
colorbar