img = imread('parrot.jpg');
img = imresize(img, 0.25);
m = size(img, 1);
n = size(img, 2);
img(m/2-5: m / 2 + 5, :, :) = 0;
img(:, n/2 - 5: n/2 + 5, :) = 0;

H = fspecial('sobel');
grad_x_1 = imfilter(img(:, :, 1), H, 'replicate', 'conv');
grad_y_1 = imfilter(img(:, :, 1), H', 'replicate', 'conv');
grad_x_2 = imfilter(img(:, :, 2), H, 'replicate', 'conv');
grad_y_2 = imfilter(img(:, :, 2), H', 'replicate', 'conv');
grad_x_3 = imfilter(img(:, :, 3), H, 'replicate', 'conv');
grad_y_3 = imfilter(img(:, :, 3), H', 'replicate', 'conv');


grad_xx = grad_x_1 .^ 2 + grad_x_2 .^ 2 + grad_x_3 .^ 2;
grad_yy = grad_y_1 .^ 2 + grad_y_2 .^ 2 + grad_y_3 .^ 2;
grad_xy = grad_x_1 .* grad_y_1 + grad_x_2 .* grad_y_2 + grad_x_3 .* grad_y_3;

grad = zeros(2,2,m*n);
grad(1,1,:) = grad_xx(:);
grad(1,2,:) = grad_xy(:);
grad(2,1,:) = grad_xy(:);
grad(2,2,:) = grad_yy(:);

% TODO Gaussian smooth structure tensor

grad_cells = num2cell(grad, [1, 2]);


[V1, D1] = cellfun(@ (x) eig(cell2mat({x})), grad_cells, 'UniformOutput', false);

V = cell2mat(V1);
D = cell2mat(D1);

V_2d = reshape(V, [2, size(V, 3) * 2]);

[~, max_index] = max(max(abs(D)));
min_index = 3 - max_index;

theta_max = V_2d(:, 2*(0:(size(V, 3) - 1)) + permute(max_index, [1, 3, 2]));
theta_min = V_2d(:, 2*(0:(size(V, 3) - 1)) + permute(min_index, [1, 3, 2]));

lambda_max = permute(max(max(abs(D))), [3, 1, 2]);

lambda_min = permute(min(max(abs(D))), [3, 1, 2]);

theta_max_2 = theta_max.^2;
theta_min_2 = theta_min.^2;
theta_max_12 = theta_max(1, :) .* theta_max(2, :);
theta_min_12 = theta_min(1, :) .* theta_min(2, :);

theta_max_in = zeros(2, 2, m * n);
theta_min_in = theta_max_in;

theta_max_in(1, 1, :) = theta_max_2(1, :);
theta_max_in(2, 2, :) = theta_max_2(2, :);
theta_max_in(1, 2, :) = theta_max_12;
theta_max_in(2, 1, :) = theta_max_12;

theta_min_in(1, 1, :) = theta_min_2(1, :);
theta_min_in(2, 2, :) = theta_min_2(2, :);
theta_min_in(1, 2, :) = theta_min_12;
theta_min_in(2, 1, :) = theta_min_12;

f_plus = 1./(1 + lambda_min + lambda_max);
f_minus = 1./(1 + sqrt(lambda_min + lambda_max));

plus_factor = reshape(repmat(repelem(f_plus', 2), [2,  1]), [2, 2, m*n]);
minus_factor = reshape(repmat(repelem(f_minus', 2), [2,  1]), [2, 2, m*n]);

T = plus_factor .* theta_max_in + minus_factor .* theta_min_in;

% TODO Eq 8, 9

