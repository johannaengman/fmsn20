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

%% Estimate beta
number_relevant = 3;

X_relevant = [ones(size(X,1),1) X(:, 2:3) min(X(:,4), X(:,5))];
X_reg = X_relevant;

beta = X_relevant\Y;

Y_pred = X_relevant*beta;
error = Y-Y_pred;

%Plot the residuals vs the covariates
% figure
% for i = 1:5
%     subplot(3,2,i)
%     plot(X(:,i),error,'.')
% end

% Estimation for the validation data
Xv_reg = [ones(size(X_valid,1),1) X_valid(:,2:3) min(X_valid(:,4), X_valid(:,5))];
Yv_pred = Xv_reg*beta;

%Calculate the estimated variance
sigma_square = (1/(numel(error)-number_relevant))*sum(error.^2);
var_of_beta = sigma_square*inv(X_reg'*X_reg);
var_Y = sum(X_reg*var_of_beta.*X_reg,2) + sigma_square;

var_Yv = sum(Xv_reg*var_of_beta.*Xv_reg, 2)+ sigma_square;

I_land = ~any(isnan(X_grid),2);
X_grid_reg = [ones(sum(I_land),1) X_grid(I_land,2:3) min(X_grid(I_land,4), X_grid(I_land,5))];

Y_grid_var = nan(sz_grid);
Y_grid_var(I_land) = sum(X_grid_reg*var_of_beta.*X_grid_reg, 2) + sigma_square;

%% Illustretes the variance
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
subplot(121)
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

subplot(122)
plot(Y_valid-Yv_pred, '.')
title('Residuals')

%% Validation

mse_ols = (1/length(Y_valid))*sum((Y_valid - Yv_pred).^2);

standard_z_ols = (Y_valid - Yv_pred)./sqrt(var_Yv);

figure
normplot(standard_z_ols)



