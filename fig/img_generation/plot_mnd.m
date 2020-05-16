mu = [0 0];
sigma = [0.3 0; 0 0.3];
k = [-2.9 -1 0.75 2.75];

x1 = -10:0.1:10;
x2 = -5:0.1:5;
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];

clrs = {'r', 'g', 'b', [1.0 0.5 0]};

for i = 1:4
    y = mvnpdf(X,[mu(1)+k(i) mu(2)],sigma);
    y = reshape(y,length(x2),length(x1));
    surf(x1,x2,y, 'FaceColor', clrs{i})
    hold on
end

caxis([min(y(:))-0.5*range(y(:)),max(y(:))])
axis([-5 5 -2.5 2.5 0 0.8])
xlabel('x [m]')
ylabel('y [m]')
zlabel('Probability Density')

title("Hypothesis probability density");