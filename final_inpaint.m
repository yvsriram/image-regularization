tic;
img = im2double(imread('chasmis.jpg'));
% img = im2double(imresize(img, 0.25));
m = size(img, 1);
n = size(img, 2);
% Parameters
t = 0.5;
sigma = 3;
w=2;
p=2;
% q=1;
% sigma1 = 1;
%

mask = rgb2gray(im2double(imread('chasma.png')));
mask(mask<=0.5) = 0;
mask(mask>0.5) = 1;
mask1 = mask;
mask2 = mask;

H = fspecial('sobel');
grad_x_1 = imfilter(img(:, :, 1), H', 'replicate', 'conv');
grad_y_1 = imfilter(img(:, :, 1), H,  'replicate', 'conv');
grad_x_2 = imfilter(img(:, :, 2), H', 'replicate', 'conv');
grad_y_2 = imfilter(img(:, :, 2), H, 'replicate', 'conv');
grad_x_3 = imfilter(img(:, :, 3), H', 'replicate', 'conv');
grad_y_3 = imfilter(img(:, :, 3), H, 'replicate', 'conv');

H = fspecial('gaussian',2*w+1,sigma);

% img1 = zeros(m+4,n+4,3);
% img1(3:m+2,3:n+2,:) = img;
% img1(3:m+2,1:2,:) = repmat(img(:,1,:),1,2,1);
% img1(3:m+2,n+3:n+4,:) = repmat(img(:,n,:),1,2,1);
% img1(1:2,3:n+2,:) = repmat(img(1,:,:),2,1,1);
% img1(m+3:m+4,3:n+2,:) = repmat(img(m,:,:),2,1,1);
% img1(1:2,1:2,:) = repmat(img(1,1,:),2,2,1);
% img1(m+3:m+4,1:2,:) = repmat(img(m,1,:),2,2,1);
% img1(1:2,n+3:n+4,:) = repmat(img(1,n,:),2,2,1);
% img1(m+3:m+4,n+3:n+4,:) = repmat(img(m,n,:),2,2,1);

G = zeros(2*p+1,2*p+1,2);
for i = -p:p
    for j = -p:p
        G(i+p+1,j+p+1,:) = [i,j];
    end
end
C = num2cell(G, 3);

while sum(mask(:)) > 0
    for i = 1:m
        for j = 1:n
            a1 = max([1-i, -1]);
            b1 = max([1-j, -1]);
            a2 = min([m-i, 1]);
            b2 = min([n-j, 1]);
            if mask(i,j) == 0 && sum(sum(mask(i+a1:i+a2,j+b1:j+b2))) > 0
                if a1==0 || mask(i+a1,j)==1
                    grad_x_1(i,j) = 8*(img(i,j,1) - img(i+a2,j,1));
                    grad_x_2(i,j) = 8*(img(i,j,2) - img(i+a2,j,2));
                    grad_x_3(i,j) = 8*(img(i,j,3) - img(i+a2,j,3));
                elseif a2==0 || mask(i+a2,j)==1
                    grad_x_1(i,j) = 8*(-img(i,j,1) + img(i+a1,j,1));
                    grad_x_2(i,j) = 8*(-img(i,j,2) + img(i+a1,j,2));
                    grad_x_3(i,j) = 8*(-img(i,j,3) + img(i+a1,j,3));
                else
                    grad_x_1(i,j) = 4*(img(i+a1,j,1) - img(i+a2,j,1));
                    grad_x_2(i,j) = 4*(img(i+a1,j,2) - img(i+a2,j,2));
                    grad_x_3(i,j) = 4*(img(i+a1,j,3) - img(i+a2,j,3));
                end
                if b1==0 || mask(i,j+b1)==1
                    grad_y_1(i,j) = 8*(img(i,j,1) - img(i,j+b2,1));
                    grad_y_2(i,j) = 8*(img(i,j,2) - img(i,j+b2,2));
                    grad_y_3(i,j) = 8*(img(i,j,3) - img(i,j+b2,3));
                elseif b2==0 || mask(i,j+b2)==1
                    grad_y_1(i,j) = 8*(-img(i,j,1) + img(i,j+b1,1));
                    grad_y_2(i,j) = 8*(-img(i,j,2) + img(i,j+b1,2));
                    grad_y_3(i,j) = 8*(-img(i,j,3) + img(i,j+b1,3));
                else
                    grad_y_1(i,j) = 4*(img(i,j+b1,1) - img(i,j+b2,1));
                    grad_y_2(i,j) = 4*(img(i,j+b1,2) - img(i,j+b2,2));
                    grad_y_3(i,j) = 4*(img(i,j+b1,3) - img(i,j+b2,3));
                end
            end
        end
    end
    
    grad_xx = grad_x_1 .^ 2 + grad_x_2 .^ 2 + grad_x_3 .^ 2;
    grad_yy = grad_y_1 .^ 2 + grad_y_2 .^ 2 + grad_y_3 .^ 2;
    grad_xy = grad_x_1 .* grad_y_1 + grad_x_2 .* grad_y_2 + grad_x_3 .* grad_y_3;

    grad = zeros(2,2,m*n);
    grad(1,1,:) = grad_xx(:);
    grad(1,2,:) = grad_xy(:);
    grad(2,1,:) = grad_xy(:);
    grad(2,2,:) = grad_yy(:);

    grad = reshape(grad,[2,2,m,n]);

    for i = 1:m
        for j = 1:n
            a1 = max([1-i, -1]);
            b1 = max([1-j, -1]);
            a2 = min([m-i, 1]);
            b2 = min([n-j, 1]);
            if mask1(i,j) == 1 && min(min(mask(i+a1:i+a2,j+b1:j+b2))) == 0
                a1 = max([1-i, -w]);
                b1 = max([1-j, -w]);
                a2 = min([m-i, w]);
                b2 = min([n-j, w]);
                H1 = H(w+1+a1:w+a2+1,w+b1+1:w+b2+1).*(1.-mask(i+a1:i+a2,j+b1:j+b2));
                denum = sum(H1(:));
                H1 = repmat(reshape(H1,1,1,a2-a1+1,b2-b1+1),2,2,1,1);
                grad(:,:,i,j) = sum(sum(H1.*grad(:,:,i+a1:i+a2,j+b1:j+b2),4),3)./denum;
                mask1(i,j) = 0;
            end
        end
    end
    
    grad = reshape(grad,[2,2,m*n]);
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

    % TODO Eq 8, 9
    T2 = num2cell(T,[1,2]);
    T1 = cellfun(@(x) inv(cell2mat({x})), T2, 'UniformOutput', false);
    T1 = reshape(cell2mat(T1),[m,n,2,2]);
    
    for i = 1:m
        for j = 1:n
            if mask(i,j) == 1 && mask1(i,j) == 0
                temp = cellfun(@(x) exp(-reshape(cell2mat({x}),[1,2])*reshape(T1(i,j,:,:),[2,2])*reshape(cell2mat({x}),[2,1])/(4*t))/(4*pi*t),C,'UniformOutput', false);
                temp = cell2mat(temp);
                a1 = max([1-i, -p]);
                b1 = max([1-j, -p]);
                a2 = min([m-i, p]);
                b2 = min([n-j, p]);
                temp = temp(p+1+a1:p+a2+1,p+b1+1:p+b2+1).*(1.-mask(i+a1:i+a2,j+b1:j+b2));
                img(i,j,:) = sum(sum(img(i+a1:i+a2,j+b1:j+b2,:).*repmat(temp,1,1,3)))./sum(temp(:));
            end
        end
    end
    mask = mask1;
end

num_iter = 2;
counter=0;

while counter < 2*num_iter
    m = size(img, 1);
    n = size(img, 2);
    % Parameters
    t = 3;
    q=2;
    sigma = 1;
    %

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

    % TODO Eq 8, 9
    T2 = num2cell(T,[1,2]);
    T1 = cellfun(@(x) inv(cell2mat({x})), T2, 'UniformOutput', false);
    T1 = reshape(cell2mat(T1),[m,n,2,2]);

    img1 = zeros(m+4,n+4,3);
    img1(3:m+2,3:n+2,:) = img;
    img1(3:m+2,1:2,:) = repmat(img(:,1,:),1,2,1);
    img1(3:m+2,n+3:n+4,:) = repmat(img(:,n,:),1,2,1);
    img1(1:2,3:n+2,:) = repmat(img(1,:,:),2,1,1);
    img1(m+3:m+4,3:n+2,:) = repmat(img(m,:,:),2,1,1);
    img1(1:2,1:2,:) = repmat(img(1,1,:),2,2,1);
    img1(m+3:m+4,1:2,:) = repmat(img(m,1,:),2,2,1);
    img1(1:2,n+3:n+4,:) = repmat(img(1,n,:),2,2,1);
    img1(m+3:m+4,n+3:n+4,:) = repmat(img(m,n,:),2,2,1);

    G = zeros(5,5,2);
    for i = -2:2
        for j = -2:2
            G(i+3,j+3,:) = [i,j];
        end
    end
    C = num2cell(G, 3);
    out_img = zeros(m,n,3);

    %out_img = img;

    for i = 1:m
        for j = 1:n
            if mask2(i,j) == 1
                temp = cellfun(@(x) exp(-reshape(cell2mat({x}),[1,2])*reshape(T1(i,j,:,:),[2,2])*reshape(cell2mat({x}),[2,1])/(4*t))/(4*pi*t),C,'UniformOutput', false);
                temp = cell2mat(temp);
                temp = temp./sum(temp(:));
                out_img(i,j,:) = sum(sum(img1(i:i+4,j:j+4,:).*repmat(temp,1,1,3)));
            else
                out_img(i,j,:) = img(i,j,:);
            end
        end
    end

    img=out_img;
    counter=counter+2;
    
end

imshow(img);
toc;