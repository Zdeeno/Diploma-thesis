%% generate data
clear all;
center = [3 3];
real_phi = 2;
real_vec = [cos(real_phi), sin(real_phi)];
k = [-2.9 -1 0.75 2.75];
var_real = 0.3;
var_big = 20;
real_samples = 5;
noise_samples = 10;
clrs = {'r', 'g', 'b', [1.0 0.5 0]};

x1 = [mvnrnd(center + k(1)*real_vec, eye(2) * var_real, real_samples); mvnrnd([0 0], var_big.*eye(2), noise_samples)];
x2 = [mvnrnd(center + k(2)*real_vec, eye(2) * var_real, real_samples); mvnrnd([0 0], var_big.*eye(2), noise_samples)];
x3 = [mvnrnd(center + k(3)*real_vec, eye(2) * var_real, real_samples); mvnrnd([0 0], var_big.*eye(2), noise_samples)];
x4 = [mvnrnd(center + k(4)*real_vec, eye(2) * var_real, real_samples); mvnrnd([0 0], var_big.*eye(2), noise_samples)];
x = {x1, x2, x3, x4};

a1 = ones(size(x1));
a2 = ones(size(x1));
a3 = ones(size(x1));
a4 = ones(size(x1));
a = {a1, a2, a3, a4};

% em algorithm
% init mean 
mean = [0 0];
phi = 0;
vec = [cos(phi), sin(phi)];
likelihood = [];

figure
for i = 1:4
    plot(mean(1) + vec(1)*k(i), mean(2) + vec(2)*k(i), "*", 'MarkerSize', 20, 'MarkerEdgeColor', clrs{i});
    hold on;
end
for i = 1:4
    plot(center(1) + real_vec(1)*k(i), center(2) + real_vec(2)*k(i), "o", 'MarkerSize', 30, 'MarkerEdgeColor', clrs{i});
    hold on;
end
for i = 1:4
    plot(x{i}(:, 1), x{i}(:, 2), ".", 'MarkerSize', 7, 'MarkerEdgeColor', clrs{i});
    hold on;
end
grid on
xlim([-10, 10]);
ylim([-10, 10]);
title("EM algorithm initialization", 'fontsize', 15);
xlabel("Distance (m)", 'fontsize', 15)
ylabel("Distance (m)", 'fontsize', 15)
sum_ax = 0;
sum_ay = 0;
m1 = 0;
m2 = 0;
for i = 1:4
    m1 = m1 + (sum(a{i}(:, 1).*x{i}(:, 1))) / (sum(a{i}(:, 1)));
    m2 = m2 + (sum(a{i}(:, 2).*x{i}(:, 2))) / (sum(a{i}(:, 2)));
end
mean(1) = m1/4;
mean(2) = m2/4;

sum_x = 0;
sum_y = 0;
% init rotation
for i = 1:4
    sum_x = sum_x + (sum( a{i}(:, 1).* (x{i}(:, 1) - mean(1)) / k(i))/sum_ax);
    sum_y = sum_y + (sum( a{i}(:, 2).* (x{i}(:, 2) - mean(2)) / k(i))/sum_ay);
end
phi = atan2(sum_y, sum_x);
vec = [cos(phi), sin(phi)];

% iterate algorithm
for epoch = 1:1000
    
    % maximize mean
    sum_x = 0;
    sum_y = 0;
    sum_ax = 0;
    sum_ay = 0;
    
    for i = 1:4
        m1 = m1 + (sum(a{i}(:, 1).*x{i}(:, 1))) / (sum(a{i}(:, 1)));
        m2 = m2 + (sum(a{i}(:, 2).*x{i}(:, 2))) / (sum(a{i}(:, 2)));
        sum_ax = sum_ax + sum(a{i}(:, 1));
        sum_ay = sum_ay + sum(a{i}(:, 2));
    end
    mean(1) = sum_x/sum_ax;
    mean(2) = sum_y/sum_ay;

    % maximize phi
    sum_x = 0;
    sum_y = 0;
    for i = 1:4
        sum_x = sum_x + (sum( a{i}(:, 1).* (x{i}(:, 1) - mean(1)) / k(i))/sum_ax);
        sum_y = sum_y + (sum( a{i}(:, 2).* (x{i}(:, 2) - mean(2)) / k(i))/sum_ay);
    end
    phi = atan2(sum_y, sum_x);
    vec = [cos(phi), sin(phi)];
    
    % expectation
    lik_sum = 0;
    for i = 1:4
        a{i}(:, 1) = normpdf(x{i}(:, 1), mean(1)+vec(1)*k(i), 1);
        a{i}(:, 2) = normpdf(x{i}(:, 2), mean(2)+vec(2)*k(i), 1);
        lik_sum = lik_sum + sum(a{i}(:, 1)) + sum(a{i}(:, 2));
    end
    likelihood = [likelihood lik_sum];
    
end

figure
for i = 1:4
    plot(mean(1) + vec(1)*k(i), mean(2) + vec(2)*k(i), "*", 'MarkerSize', 25, 'MarkerEdgeColor', clrs{i});
    hold on;
end
for i = 1:4
    plot(center(1) + real_vec(1)*k(i), center(2) + real_vec(2)*k(i), "o", 'MarkerSize', 35, 'MarkerEdgeColor', clrs{i});
    hold on;
end
for i = 1:4
    plot(x{i}(:, 1), x{i}(:, 2), ".", 'MarkerSize', 9, 'MarkerEdgeColor', clrs{i});
    hold on;
end
grid on
xlim([-2, 8]);
ylim([-2, 8]);
title("EM algorithm result", 'fontsize', 15);
xlabel("Distance (m)", 'fontsize', 15)
ylabel("Distance (m)", 'fontsize', 15)

mean
phi

% figure
% plot(likelihood, "LineWidth", 1);
% title("Evolution of likelihood");
% xlabel("Iterations");
% ylabel("Likelihood");
% grid on;

%% convergence
clear all;
center = [0 0];
real_phi = 2;
real_vec = [cos(real_phi), sin(real_phi)];
k = [-2.9 -1 0.75 2.75];
var_real = 0.3;
var_big = 20;
real_samples = 5;
clrs = {'r', 'g', 'b', [1.0 0.5 0]};

x1 = mvnrnd(center + k(1)*real_vec, eye(2) * var_real, real_samples);
x2 = mvnrnd(center + k(2)*real_vec, eye(2) * var_real, real_samples);
x3 = mvnrnd(center + k(3)*real_vec, eye(2) * var_real, real_samples);
x4 = mvnrnd(center + k(4)*real_vec, eye(2) * var_real, real_samples);
x = {x1, x2, x3, x4};

mean = [0, 0];
phi = -pi:0.01:pi;
% expectation

likelihood = zeros(1, numel(phi));
for step = 1:numel(phi)
    vec = [cos(phi(step)), sin(phi(step))];
    lik_sum = 0;
    for i = 1:4
        a{i}(:, 1) = normpdf(x{i}(:, 1), mean(1)+vec(1)*k(i), 1);
        a{i}(:, 2) = normpdf(x{i}(:, 2), mean(2)+vec(2)*k(i), 1);
        lik_sum = lik_sum + sum(a{i}(:, 1)) + sum(a{i}(:, 2));
    end
    likelihood(step) = lik_sum;
end

phi
likelihood
plot(phi, likelihood, "LineWidth", 2)
grid on
title("Likelihood with respect to \phi", 'fontsize', 13)
xlabel("\phi (rad)", 'fontsize', 13)
ylabel("Likelihood", 'fontsize', 13)


