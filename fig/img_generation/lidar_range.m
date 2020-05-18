%% horizontal
clear all;
b = 0:0.01:20;
alpha = 0.1;
a = 0.3;
alpha = deg2rad(alpha);
N = acos(1 - (a.^2 ./ (2.*b.^2)))./alpha;
plot(b, N, "linewidth", 1.5);
grid on;
xlim([2 20]);
xlabel("Distance [m]", 'fontsize', 13);
ylabel("Number of hits", 'fontsize', 13);
title("Horizontal range analysis", 'fontsize', 13);
legend({"hits"}, 'fontsize', 13);


%% vertical
clear all;
b = 0:0.01:10;
alpha = 2;
a = 0.4;
alpha = deg2rad(alpha);
N = acos(1 - (a.^2 ./ (2.*b.^2)))./alpha;
plot(b, N, "linewidth", 1.5);
grid on;
xlim([1 10]);
xlabel("Distance [m]", 'fontsize', 13);
ylabel("Number of hits", 'fontsize', 13);
title("Vertical range analysis", 'fontsize', 13);
legend({"hits"}, 'fontsize', 13);
