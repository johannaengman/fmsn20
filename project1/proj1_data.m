%set paths to course files and download UStemp.mat from the homepage
%% load data
load UStemp.mat

%% plot observations
figure(1)
subplot(331)
scatter(X(:,1), X(:,2), 20, Y, 'filled')
axis xy tight; hold off; colorbar
title('Winter temperature'); xlabel('longitude'); ylabel('latitude')

%plot long+lat+elevation (prediction surface + at observations sites)
for i=1:5
  subplot(3, 3, i+1)
  imagesc([min(X_grid(:,1)) max(X_grid(:,1))], ...
    [max(X_grid(:,2)) min(X_grid(:,2))], ...
    reshape(X_grid(:,i),sz_grid), ...
    'alphadata', ~isnan(reshape(X_grid(:,3),sz_grid)))
  hold on
  scatter(X(:,1), X(:,2), 20, X(:,i), ...
    'filled','markeredgecolor','k')
  axis xy tight; hold off; colorbar
  title(names{i})
end
subplot(3, 3, 7)
imagesc([min(X_grid(:,1)) max(X_grid(:,1))], ...
	[max(X_grid(:,2)) min(X_grid(:,2))], ...
	reshape(min(X_grid(:,4),X_grid(:,5)),sz_grid), ...
	'alphadata', ~isnan(reshape(X_grid(:,3),sz_grid)))
hold on
scatter(X(:,1), X(:,2), 20, min(X(:,4),X(:,5)), ...
	'filled','markeredgecolor','k')
axis xy tight; hold off; colorbar
title('distance to any coast')

figure
imagesc([min(X_grid(:,1)) max(X_grid(:,1))], ...
    [max(X_grid(:,2)) min(X_grid(:,2))], ...
    reshape(X_grid(:,3),sz_grid), ...
    'alphadata', ~isnan(reshape(X_grid(:,3),sz_grid)))
hold on
scatter(X(:,1), X(:,2), 20, X(:,3), ...
    'filled','markeredgecolor','k')
axis xy tight; hold off; colorbar
title(names{3})

%% perform a linear regression on latitude and elevation
X_reg = [ones(size(Y,1),1) X(:,2:3)];
[b, bint] = regress(Y, X_reg);

%% use regression to predict at observation sites
Y_pred = X_reg*b;
%at validation sites
Xv_reg = [ones(size(X_valid,1),1) X_valid(:,2:3)];
Yv_pred = Xv_reg*b;
%and at the grid locations. Note that we need to handle the NaN values
%marking water.
I_land = ~any(isnan(X_grid),2);
X_grid_reg = [ones(sum(I_land),1) X_grid(I_land,2:3)];
%extract covariates (elevation) for the prediction locations
Y_grid = nan(sz_grid);
Y_grid(I_land) = X_grid_reg*b;

%% and plot the results
%First plot the gridded reconstructions along with observations
figure(2)
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
title('Gridded predictions')

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
title('Gridded predictions')

%and residuals
subplot(223)
scatter(X(:,1), X(:,2), 20, Y-Y_pred, ...
  'filled','markeredgecolor','k')
hold on
scatter(X_valid(:,1), X_valid(:,2), 20, Y_valid-Yv_pred, ...
  'filled','markeredgecolor','r')

subplot(224)
plot(Y-Y_pred, '.')
