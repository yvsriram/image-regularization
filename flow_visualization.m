close all;
clear;
tic;
img = double(imread('chasmis.jpg'));
img_old=img;
flow = 1;


num_iter = 50;
counter=0;
m = size(img, 1);
n = size(img, 2);

% Parameters
if flow == 1
    dt = 1;
else
    dt = 2;
end

q=2;
sigma = 1;

[Y, X] = meshgrid(1:n, 1:m);

if flow == 1

    Gx = (sign(X - m / 2));
    Gy = (sign(Y - n / 2));

else
    
    Gx(Y<n/2) = 0;
    Gy(Y<n/2) = -1;
    Gx(Y>=n/2) = -1;
    Gy(Y>=n/2) = 0;
end

figure;
quiver(downsample(downsample(reshape(Gx, m, n), 30)', 30)', downsample(downsample(reshape(Gy, m, n), 30)', 30)');
title('Flow vector field')

Gx = Gx(:);
Gy = Gy(:);
H = zeros(2, 2, m * n);
H(1, 1, :) = Gx.^2 ./ ((Gx.^2 + Gy.^2));
H(2, 2, :) = Gy.^2 ./ ((Gx.^2 + Gy.^2));
H(1, 2, :) = Gx.*Gy ./ ((Gx.^2 + Gy.^2));

s = size(intersect(find(Gx == 0), find(Gy == 0)), 1); 
H(2, 1, :) = H(1, 2, :);
if  s > 0 
    H(:,:, intersect(find(Gx == 0), find(Gy == 0))) = zeros(2, 2, s);  
end

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

    
    
    M_r = arrayfun(@(x) trace(reshape(H(:,:,x), [2, 2]) * reshape(hessianr(:, :, x), [2, 2])), 1:m*n);
    M_g = arrayfun(@(x) trace(reshape(H(:,:,x), [2, 2]) * reshape(hessiang(:, :, x), [2, 2])), 1:m*n);
    M_b = arrayfun(@(x) trace(reshape(H(:,:,x), [2, 2]) * reshape(hessianb(:, :, x), [2, 2])), 1:m*n);
    
    img(:, :, 1) = img(:, :, 1) + dt * reshape(M_r, m, n);
    img(:, :, 2) = img(:, :, 2) + dt * reshape(M_g, m, n);
    img(:, :, 3) = img(:, :, 3) + dt * reshape(M_b, m, n);
    img(img<0) = 0;
    counter=counter+2;
    if counter == 50
        figure;
        imshow(uint8(img));
        title 'Smoothing in direction of flow: 25 iterations'
    end
end
figure;
imshow(uint8(img));
title 'Smoothing in direction of flow: 50 iterations'
toc; 