clear all;
clc;
x1 = [1 -1; 0.5 0; -1 -1];
x2 = [10 10.5; 9.5 9.5; 11 9]; 
alphas = [1 1 1 1 1 1];
mean = [10, 0];
dist = sqrt(50);
rot = 0;

plot(x1(:, 1), x1(:, 2), "o",'MarkerEdgeColor', "r");
hold on;
plot(x2(:, 1), x2(:, 2), "o",'MarkerEdgeColor', "b");
grid on;
hold on;

% Max likelihood
mean(1) = (sum(x1(:, 1) .* alphas(1:3)') + sum(x2(:, 1) .* alphas(4:6)'))/numel(alphas);
mean(2) = (sum(x1(:, 2) .* alphas(1:3)') + sum(x2(:, 2) .* alphas(4:6)'))/numel(alphas);

upper_sum = sum(alphas(1:3)' .*  ((x1(:, 2) - mean(2))./(dist)));
upper_sum = upper_sum + sum(alphas(4:6)' .*  ((x2(:, 2) - mean(2))./-dist))
bott_sum = sum(alphas(1:3)' .*  ((x1(:, 1) - mean(1))./(dist)));
bott_sum = bott_sum + sum(alphas(4:6)' .*  ((x2(:, 1) - mean(1))./-dist))
rot = atan2(upper_sum, bott_sum);

disp(mean)
disp(rad2deg(rot))

vec = [cos(rot) sin(rot)] * dist;
plot(mean(1) + vec(1), mean(2) + vec(2), "o", 'MarkerSize', 50, 'MarkerEdgeColor','r');
hold on;
plot(mean(1) - vec(1), mean(2) - vec(2), "o", 'MarkerSize', 50, 'MarkerEdgeColor','b');




