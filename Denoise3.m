tic;
img = im2double(imread('noisybaboon.png'));
%img = im2double(imresize(img, 0.25));
img_old=img;
img=img + randn(size(img))*0.05;

num_iter = 4;
counter=0;
m = size(img, 1);
n = size(img, 2);

% Parameters
dt = 1;
q=2;
sigma = 1;  

while counter < 2*num_iter
   
    [Igxr, Igyr] = gradient(img(:, :, 1));
    [Igxxr, Igyxr] = gradient(Igxr);
    [Igxyr, Igyyr] = gradient(Igyr);
    
    
    [Igxg, Igyg] = gradient(img(:, :, 2));
    [Igxxg, Igyxg] = gradient(Igxg);
    [Igxyg, Igyyg] = gradient(Igyg);
    
    
    [Igxb, Igyb] = gradient(img(:, :, 3));
    [Igxxb, Igyxb] = gradient(Igxb);
    [Igxyb, Igyyb] = gradient(Igyb);
    
    hessianr = zeros(2, 2, m*n);
    hessianr(1, 1, :) = Igxxr(:);
    hessianr(2, 2, :) = Igyyr(:);
    hessianr(1, 2, :) = Igxyr(:);
    hessianr(2, 1, :) = Igyxr(:);
    
    
    hessiang = zeros(2, 2, m*n);
    hessiang(1, 1, :) = Igxxg(:);
    hessiang(2, 2, :) = Igyyg(:);
    hessiang(1, 2, :) = Igxyg(:);
    hessiang(2, 1, :) = Igyxg(:);
    
    
    hessianb = zeros(2, 2, m*n);
    hessianb(1, 1, :) = Igxxb(:);
    hessianb(2, 2, :) = Igyyb(:);
    hessianb(1, 2, :) = Igxyb(:);
    hessianb(2, 1, :) = Igyxb(:);

    H = fspecial('sobel');
    grad_x_1 = imfilter(img(:, :, 1), H', 'replicate', 'conv');
    grad_y_1 = imfilter(img(:, :, 1), H, 'replicate', 'conv');
    grad_x_2 = imfilter(img(:, :, 2), H', 'replicate', 'conv');
    grad_y_2 = imfilter(img(:, :, 2), H, 'replicate', 'conv');
    grad_x_3 = imfilter(img(:, :, 3), H', 'replicate', 'conv');
    grad_y_3 = imfilter(img(:, :, 3), H, 'replicate', 'conv');


    grad_xx = grad_x_1 .^ 2 + grad_x_2 .^ 2 + grad_x_3 .^ 2;
    grad_yy = grad_y_1 .^ 2 + grad_y_2 .^ 2 + grad_y_3 .^ 2;
    grad_xy = grad_x_1 .* grad_y_1 + grad_x_2 .* grad_y_2 + grad_x_3 .* grad_y_3;

    h = fspecial('gaussian',2*q+1,sigma);
    
    grad_xx = imfilter(grad_xx,h,'replicate', 'conv');
    grad_yy = imfilter(grad_yy,h,'replicate', 'conv');
    grad_xy = imfilter(grad_xy,h,'replicate', 'conv');

    grad = zeros(2,2,m*n);
    grad(1,1,:) = grad_xx(:);
    grad(1,2,:) = grad_xy(:);
    grad(2,1,:) = grad_xy(:);
    grad(2,2,:) = grad_yy(:);

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
    f_minus = 1./sqrt(1 + lambda_min + lambda_max);

    plus_factor = reshape(repmat(repelem(f_plus', 2), [2,  1]), [2, 2, m*n]);
    minus_factor = reshape(repmat(repelem(f_minus', 2), [2,  1]), [2, 2, m*n]);

    T = plus_factor .* theta_max_in + minus_factor .* theta_min_in;
   
    M_r = arrayfun(@(x) trace(reshape(T(:,:,x), [2, 2]) * reshape(hessianr(:, :, x), [2, 2])), 1:m*n);
    M_g = arrayfun(@(x) trace(reshape(T(:,:,x), [2, 2]) * reshape(hessiang(:, :, x), [2, 2])), 1:m*n);
    M_b = arrayfun(@(x) trace(reshape(T(:,:,x), [2, 2]) * reshape(hessianb(:, :, x), [2, 2])), 1:m*n);
    
    img(:, :, 1) = img(:, :, 1) + dt * reshape(M_r, m, n);
    img(:, :, 2) = img(:, :, 2) + dt * reshape(M_g, m, n);
    img(:, :, 3) = img(:, :, 3) + dt * reshape(M_b, m, n);
    img(img<0) = 0;
    counter=counter+2;
    figure;
    imshow(img);
    toc;
    tic
end

imshow(img);

toc; 