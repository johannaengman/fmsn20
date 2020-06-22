%% Kriging
ols
close all
%Permutation
D = distance_matrix(X(:, 1:2));
Dmax = 30;
bins = 50;
[rhat_org, ~, ~, ~, d] = covest_nonparametric(D, error, bins, Dmax);

rhat_all = zeros(100, bins+1);
for i = 1:100
   res_test = error(randperm(length(error)));
   [rhat_all(i,:), ~, ~, ~, d] = covest_nonparametric(D, res_test, bins, Dmax);
end

%% Check if significant dependence in residuals
quant = quantile(rhat_all, [0.025, 0.975]);

figure
plot(d, quant)
hold on 
plot(d, rhat_org)

%% Estimates the parameters for Kriging

[par,beta_krig] = covest_ml(D, Y, 'gaussian', [], X_relevant);

%% Parametric covariance
r = gaussian_covariance(d, par(1), par(2));

%% Plot the non-parametric and parametric covaraince
figure
plot(d, quant)
hold on 
plot(d, rhat_org)
plot(d, r)

%% Calculates the estimates
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
%% Plot the estimate
I_land = ~any(isnan(X_grid),2);
%X_grid_reg = [ones(sum(I_land),1) X_grid(I_land,2:3) min(X_grid(I_land,4), X_grid(I_land,5))];

Y_grid_krig = nan(sz_grid);
Y_grid_krig(I_land) = E_grid;

%röda - valideringsdatan
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
title('Predictions based on the measured values.')

subplot(122)
plot(Y_valid-E_valid, '.')
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
title('Linear regression')


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
title('Variance of the reconstruction')

%% Validation

mse_kriging = (1/length(Y_valid))*sum((Y_valid-E_valid).^2);

standard_z = (Y_valid - E_valid)./sqrt(var_krig(1:100));

normplot(standard_z)

res = Y_valid - E_valid;

%% Validation of prediction using confidence interval
alpha_linear = 1.95;
alpha_kriging = 1.95;

Y_upper_linear = Yv_pred + alpha_linear*sqrt(var_Yv); 
Y_lower_linear = Yv_pred - alpha_linear*sqrt(var_Yv);


Y_upper_kriging = E_valid + alpha_kriging*sqrt(var_krig(1:100));
Y_lower_kriging = E_valid - alpha_kriging*sqrt(var_krig(1:100));

number_outliers_linear = 0;
for i=1:length(Y_valid)
    outliers_linear = Y_valid(i)<Y_lower_linear(i)| Y_valid(i)>Y_upper_linear(i);
    if outliers_linear
        number_outliers_linear = number_outliers_linear+1;
    end
end

number_outliers_kriging = 0;
for i=1:length(Y_valid)
    outliers_kriging = Y_valid(i)<Y_lower_kriging(i)| Y_valid(i)>Y_upper_kriging(i);
    if outliers_kriging
        number_outliers_kriging = number_outliers_kriging+1;
    end
end

figure
plot(Y_upper_linear, 'LineWidth',2)
hold on
plot(Y_lower_linear, 'LineWidth',2)
plot(Y_valid,'b.', 'MarkerSize', 12)
legend('upper quantile', 'lower quantile', 'validation data')
title('Validation of prediction using OLS with 95% confidence interval')

figure
plot(Y_upper_kriging, 'LineWidth',2)
hold on
plot(Y_lower_kriging, 'LineWidth',2)
plot(Y_valid,'b.', 'MarkerSize', 12)
legend('upper quantile', 'lower quantile', 'validation data')
title('Validation of prediction using Kriging with 95% confidence interval')

