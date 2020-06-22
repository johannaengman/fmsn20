%1. Estimates of all relevant parameters
%2. Reconstructed fields for the locations in X grid.
%3. Uncertainties for the reconstructed fields. (Prediction intervalls or standard
%   deviations for the regression and Kriging)
%4. Some evaluation of model fit using the validation data in Y valid.
%5. A comparisson of the two methods.
clc
clear
close all
load('UStemp.mat')

%% Ordinary least squares

%Plot the variates
figure
subplot(3,2,1)
plot(X(:,1), Y, '.')
title('Longitude for each location')
subplot(3,2,2)
plot(X(:,2), Y, '.')
title('Latitude for each location')
subplot(3,2,3)
plot(X(:,3), Y, '.')
title('Elevation in meters')
subplot(3,2,4)
plot(X(:,4), Y, '.')
title('Distance to the east cost in degrees')
subplot(3,2,5)
plot(X(:,5), Y, '.')
title('Distance to the west cost in degrees')
%%
figure
plot(min(X(:,4),X(:,5)), Y, '.')

%% Estimate beta
number_relevant = 3;

X_relevant = [ones(size(X,1),1) X(:, 2:3) min(X(:,4), X(:,5))];
X_reg = [ones(size(X,1),1) X(:,2:3) min(X(:,4), X(:,5))];

beta = X_relevant\Y;

Y_pred = X_relevant*beta;
error = Y-Y_pred; %Residual


%Plot the residuals vs the covariates
figure
for i = 1:5
    subplot(3,2,i)
    plot(X(:,i),error,'.')
end

Xv_reg = [ones(size(X_valid,1),1) X_valid(:,2:3) min(X_valid(:,4), X_valid(:,5))];
Yv_pred = Xv_reg*beta;

%Osäkerheter = V(Y_pred)= V(X*beta + epsilon) = V(X*beta) + V(epsilon)

%Calculates the estimates variance
sigma_square = (1/(numel(error)-number_relevant))*sum(error.^2);
var_of_beta = sigma_square*inv(X_reg'*X_reg);
var_Y = sum(X_reg*var_of_beta.*X_reg,2) + sigma_square;

var_Yv = sum(Xv_reg*var_of_beta.*Xv_reg, 2)+ sigma_square;

I_land = ~any(isnan(X_grid),2);
X_grid_reg = [ones(sum(I_land),1) X_grid(I_land,2:3) min(X_grid(I_land,4), X_grid(I_land,5))];

Y_grid_var = nan(sz_grid);
Y_grid_var(I_land) = sum(X_grid_reg*var_of_beta.*X_grid_reg, 2) + sigma_square;

%% Hur ska vi illustrera variansen?
figure
imagesc([min(X_grid(:,1)) max(X_grid(:,1))], ...
  [max(X_grid(:,2)) min(X_grid(:,2))], Y_grid_var, ...
  'alphadata', reshape(I_land,sz_grid))
hold on
scatter(X(:,1), X(:,2), 20, var_Y, ...
  'filled','markeredgecolor','k')
scatter(X_valid(:,1), X_valid(:,2), 20, var_Yv, ...
  'filled','markeredgecolor','r')
axis xy tight; hold off; colorbar
title('Variance of the reconstruction.')

%% Plots the grids based on the measured values, the predicted values
I_land = ~any(isnan(X_grid),2);
X_grid_reg = [ones(sum(I_land),1) X_grid(I_land,2:3) min(X_grid(I_land,4), X_grid(I_land,5))];

Y_grid = nan(sz_grid);
Y_grid(I_land) = X_grid_reg*beta;

%röda - valideringsdatan
figure
subplot(221)
imagesc([min(X_grid(:,1)) max(X_grid(:,1))], ...
  [max(X_grid(:,2)) min(X_grid(:,2))], Y_grid, ...
  'alphadata', reshape(I_land,sz_grid))
hold on
scatter(X(:,1), X(:,2), 20, Y, ...
  'filled','markeredgecolor','k')
scatter(X_valid(:,1), X_valid(:,2), 20, Y_valid, ...
  'filled','markeredgecolor','r')
axis xy tight; hold off; colorbar
title('Predictions based on the measured values.')

%then the gridded reconstructions and predictions
subplot(222)
imagesc([min(X_grid(:,1)) max(X_grid(:,1))], ...
  [max(X_grid(:,2)) min(X_grid(:,2))], Y_grid, ...
  'alphadata', reshape(I_land,sz_grid))
hold on
scatter(X(:,1), X(:,2), 20, Y_pred, ...
  'filled','markeredgecolor','k')
scatter(X_valid(:,1), X_valid(:,2), 20, Yv_pred, ...
  'filled','markeredgecolor','r')
axis xy tight; hold off; colorbar
title('Predictions based on the reconstructed measure values.')

%and residuals
subplot(223)
scatter(X(:,1), X(:,2), 20, Y-Y_pred, ...
  'filled','markeredgecolor','k')
hold on
scatter(X_valid(:,1), X_valid(:,2), 20, Y_valid-Yv_pred, ...
  'filled','markeredgecolor','r')
colorbar
title('Residuals, both for meassured and validation points.')

subplot(224)
plot(Y-Y_pred, '.')
title('Residuals')


%% Kriging

%Permutation

D = distance_matrix(X(:, 1:2));
%Dmax = max(max(D));
Dmax = 30;
bins = 50;
[rhat_org, ~, ~, ~, d] = covest_nonparametric(D, error, bins, Dmax);

rhat_all = zeros(100, bins+1);
for i = 1:100
   res_test = error(randperm(length(error)));
   [rhat_all(i,:), ~, ~, ~, d] = covest_nonparametric(D, res_test, bins, Dmax);
end

%%
quant = quantile(rhat_all, [0.025, 0.975]);

figure
plot(d, quant)
hold on 
plot(d, rhat_org)

%Verkar vara signifikant

[par,beta_krig] = covest_ml(D, Y, 'gaussian', [], X_relevant);

%%
X_all = [ones(size(X,1),1) X(:, 2:3) min(X(:,4), X(:,5)); ...
  ones(size(X_valid,1),1) X_valid(:, 2:3) min(X_valid(:,4), X_valid(:,5));...
  ones(sum(I_land),1) X_grid(I_land, 2:3) min(X_grid(I_land,4), X_grid(I_land,5))]; %Vilka X ska vi ha?
X_all_coords = [X; X_valid; X_grid(I_land,:)];
X_u = [X_valid; X_grid(I_land,:)];
I_obs = false(size(X_all, 1), 1);
I_obs(1:500, :) = true;

D_all = distance_matrix(X_all_coords(:, 1:2));

%Sigma = matern_covariance(D_all, par(1),par(2), par(3));
%Sigma = exponential_covariance(D_all, par(1), par(2));
Sigma = gaussian_covariance(D_all, par(1), par(2));
%Sigma = spherical_covariance(D_all, par(1), par(2));
%Sigma = cauchy_covariance(D_all, par(1), par(2), par(3));
sigma2 = par(end);
Sigma_yy = Sigma + sigma2*eye(size(Sigma)); %Kvadrat?
%%
Sigma_uu = Sigma_yy(~I_obs, ~I_obs);
Sigma_uo = Sigma_yy(~I_obs, I_obs);
Sigma_oo = Sigma_yy(I_obs, I_obs);
%y_o = y(I_obs);
%y_u = y(~I_obs);

muu = X_all(501:end,:)*beta_krig;
muo = X_all(1:500,:)*beta_krig;
Eyuyo = muu + Sigma_uo*(Sigma_oo\(Y-muo));
E_valid = Eyuyo(1:size(X_valid,1));
E_grid = Eyuyo((size(X_valid,1)+1):end);
%%
I_land = ~any(isnan(X_grid),2);
%X_grid_reg = [ones(sum(I_land),1) X_grid(I_land,2:3) min(X_grid(I_land,4), X_grid(I_land,5))];

Y_grid_krig = nan(sz_grid);
Y_grid_krig(I_land) = E_grid;

%röda - valideringsdatan
figure
subplot(221)
imagesc([min(X_grid(:,1)) max(X_grid(:,1))], ...
  [max(X_grid(:,2)) min(X_grid(:,2))], Y_grid_krig, ...
  'alphadata', reshape(I_land,sz_grid))
hold on
scatter(X(:,1), X(:,2), 20, Y, ...
  'filled','markeredgecolor','k')
scatter(X_valid(:,1), X_valid(:,2), 20, Y_valid, ...
  'filled','markeredgecolor','r')
axis xy tight; hold off; colorbar
title('Gridded predictions based on the measured values')

%then the gridded reconstructions and predictions
subplot(222)
imagesc([min(X_grid(:,1)) max(X_grid(:,1))], ...
  [max(X_grid(:,2)) min(X_grid(:,2))], Y_grid_krig, ...
  'alphadata', reshape(I_land,sz_grid))
hold on
scatter(X(:,1), X(:,2), 20, Y_pred, ...
  'filled','markeredgecolor','k')
scatter(X_valid(:,1), X_valid(:,2), 20, E_valid, ...
  'filled','markeredgecolor','r')
axis xy tight; hold off; colorbar
title('Gridded predictions besaed on our beta')

%and residuals
subplot(223)
scatter(X(:,1), X(:,2), 20, Y-Y_pred, ...
  'filled','markeredgecolor','k')
hold on
scatter(X_valid(:,1), X_valid(:,2), 20, Y_valid-Yv_pred, ...
  'filled','markeredgecolor','r')
colorbar
title('Något')

subplot(224)
plot(Y-Y_pred, '.')
title('Residuals')

%% Compairison between OLS and Kriging

figure
subplot(121)
imagesc([min(X_grid(:,1)) max(X_grid(:,1))], ...
  [max(X_grid(:,2)) min(X_grid(:,2))], Y_grid_krig, ...
  'alphadata', reshape(I_land,sz_grid))
hold on
scatter(X(:,1), X(:,2), 20, Y, ...
  'filled','markeredgecolor','k')
scatter(X_valid(:,1), X_valid(:,2), 20, Y_valid, ...
  'filled','markeredgecolor','r')
axis xy tight; hold off; colorbar
title('Kriging')

subplot(122)
imagesc([min(X_grid(:,1)) max(X_grid(:,1))], ...
  [max(X_grid(:,2)) min(X_grid(:,2))], Y_grid, ...
  'alphadata', reshape(I_land,sz_grid))
hold on
scatter(X(:,1), X(:,2), 20, Y, ...
  'filled','markeredgecolor','k')
scatter(X_valid(:,1), X_valid(:,2), 20, Y_valid, ...
  'filled','markeredgecolor','r')
axis xy tight; hold off; colorbar
title('OLS')


%% Variance kriging

var_kriging = Sigma_uu - Sigma_uo*(Sigma_oo\Sigma_uo')...
    + (X_u' - X'*(Sigma_oo\Sigma_uo'))'*((X'*(Sigma_oo\X))\...
    (X_u' - X'*(Sigma_oo\Sigma_uo')));

var_krig = diag(var_kriging);

Y_grid_krig_var = nan(sz_grid);
Y_grid_krig_var(I_land) = var_krig((size(X_valid,1)+1):end);

figure

imagesc([min(X_grid(:,1)) max(X_grid(:,1))], ...
  [max(X_grid(:,2)) min(X_grid(:,2))], Y_grid_krig_var, ...
  'alphadata', reshape(I_land,sz_grid))
hold on
scatter(X_valid(:,1), X_valid(:,2), 20, var_krig(1:100), ...
  'filled','markeredgecolor','r')
axis xy tight; hold off; colorbar
title('Kriging')





