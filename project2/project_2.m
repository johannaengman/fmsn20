load('HA2_Brazil.mat')

%extract observations, E and covariates
Y = Insurance(:,2);
E = Insurance(:,1);
%B = Insurance(:, 4:end);
B = [Insurance(:,5) Insurance(:,7) Insurance(:,8) Insurance(:,10)];

%observation matrix for all locations
A = speye(length(Y));
%find missing (nan) observations
I = ~isnan(Y);

%we need a global variable for x_mode to reuse it
%between optimisation calls
global x_mode;
x_mode = [];

%attempt to estimate parameters (optim in optim...)
%subset to only observed points here
par0 = [0, 0]; % [0, 0]

error_term = true;
isCAR = false;

par = fminsearch( @(x) gmrf_negloglike_NG(x, Y(I), A(I, :), B(I,:),...
    G, E(I), error_term, isCAR), par0);

%conditional mean is now given be the mode
E_xy = x_mode;
if error_term
    z = [A A B]*x_mode;
else
    z = [A B]*x_mode;
end

%% 
tau = exp(par(1));
q_e = exp(par(2));
n = size(G,1);

if isCAR
    Q_x = tau*G;
else
    Q_x = tau*G*G;
end
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

[~, ~, Q_xy] = gmrf_taylor(E_xy, Y(I), Aall(I, :), Qall, E(I), par);
    
e = [zeros(size(Q_xy,1)-size(B,2), size(B,2)); eye(size(B,2))];
V_beta0 = e'*(Q_xy\e);

%%  Check so that the covariates actually are significant.
beta = E_xy(end-size(B,2)+1:end); %Last 9 values of E_xy
conf = 2*sqrt(diag(V_beta0));
beta_min = beta-conf;
beta_max = beta+conf;

figure
plot(beta_min, 'o-')
hold on
plot(beta_max, 'o-')
plot(diag(V_beta0), 'o')


%% Calculating the expected value of the reconstructed data z

if error_term
    z_mean = [zeros(size(A)) zeros(size(A)) B]*x_mode;
    z_smooth = [A zeros(size(A,1)) zeros(size(B))]*x_mode;
    z_complete = [A zeros(size(A)) B]*x_mode;
    z_noise = [zeros(size(A)) A zeros(size(B))]*x_mode;
else
    z_mean = [zeros(size(A)) B]*x_mode;
    z_smooth = [A zeros(size(B))]*x_mode;
    z_complete = [A B]*x_mode;
end

%% Calculating the variance of the reconstructed data z for the different components
    
R_xy = chol(Q_xy);
e = randn(size(R_xy,1),1000); %random gaussian noise
x_samp = E_xy + R_xy\e;

if error_term
    z_samp_mean = [zeros(size(A)) zeros(size(A)) B]*x_samp;
    z_samp_smooth = [A zeros(size(A,1)) zeros(size(B))]*x_samp;
    z_samp_complete = [A zeros(size(A)) B]*x_samp;
    z_samp_noise = [zeros(size(A)) A zeros(size(B))]*x_samp;
    
    var_z_noise = var(z_samp_noise');
    std_z_noise = std(z_samp_noise');
else
    z_samp_mean = [zeros(size(A)) B]*x_samp;
    z_samp_smooth = [A zeros(size(B))]*x_samp;
    z_samp_complete = [A B]*x_samp;
end

var_z_mean = var(z_samp_mean');
var_z_smooth = var(z_samp_smooth');
var_z_complete = var(z_samp_complete');

std_z_mean = std(z_samp_mean');
std_z_smooth = std(z_samp_smooth');
std_z_complete = std(z_samp_complete');

%% Mean of variance och standard deviation
if error_term
    mean_var_z_noise_error = mean(var_z_noise);
    mean_std_z_noise_error = mean(std_z_noise);
end

mean_var_z_mean_error = mean(var_z_mean);
mean_var_z_smooth_error = mean(var_z_smooth);
mean_var_z_complete_error = mean(var_z_complete);

mean_std_z_mean = mean(std_z_mean);
mean_std_z_smooth = mean(std_z_smooth);
mean_std_z_complete = mean(std_z_complete);

%% Plot the variance of the reconstructed data z for the different components
figure
subplot(2,2,1)
plotMap(BrazilMap, var_z_complete, 'none')
lim = caxis;
colorbar
axis tight
title('Variance of reconstructed data with all the spatial structure component.')
subplot(2,2,2)
plotMap(BrazilMap, var_z_mean, 'none')
caxis(lim)
colorbar
axis tight
title('Variance of reconstructed data with the mean component.')
subplot(2,2,3)
plotMap(BrazilMap, var_z_smooth, 'none')
caxis(lim)
colorbar
axis tight
title('Variance of reconstructed data with the spacial smooth component.')
if error_term
    subplot(2,2,4)
    plotMap(BrazilMap, var_z_noise, 'none')
    caxis(lim)
    colorbar
    axis tight
    title('Variance of reconstructed data with the noise component.')
end
%%
log_risk = log(Y./E);

%%
figure
plotMap(BrazilMap, log_risk, 'none')
colorbar
axis tight
title('log risk')
lim_log_risk = caxis;

figure
plotMap(BrazilMap, z, 'none')
caxis(lim_log_risk)
colorbar
axis tight
title('log risk')

%%
figure
subplot(2, 2, 1)
plotMap(BrazilMap, z_samp_complete, 'none')
colorbar
axis tight
title('Complete')
lim_e = caxis;
subplot(2, 2, 2)
plotMap(BrazilMap, z_samp_mean, 'none')
caxis(lim_e)
colorbar
axis tight
title('Mean model')
subplot(2, 2, 3)
plotMap(BrazilMap, z_samp_smooth, 'none')
caxis(lim_e)
colorbar
axis tight
title('Smooth')
if error_term
    subplot(2, 2, 4)
    plotMap(BrazilMap, z_samp_noise, 'none')
    caxis(lim_e)
    colorbar
    axis tight
    title('Noise')
end