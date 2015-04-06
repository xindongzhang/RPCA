clear all;
close all;
clc;
%% 
addpath ./data;


% open the movie
n_frames = 86;
img = imread('1.jpg');
img = rgb2gray(img);
[height, width] = size(img);
% vectorize every frame to form matrix X
X = zeros(n_frames, height*width);
for i = (1:n_frames)
    file = strcat(num2str(i), '.jpg');
    frame = imread(file);
    frame = rgb2gray(frame);
    X(i,:) = reshape(frame,[],1);
end

% apply Robust PCA
lambda = 1/sqrt(max(size(X)));
tic
[L,S] = RPCA(X, lambda/3, 10*lambda/3, 1e-5, 1000);
toc
frame1 = reshape(X(86,:),height,[]);
frame2 = reshape(L(86,:),height,[]);
frame3 = reshape(abs(S(86,:)),height,[]);
frame1 = uint8(frame1);
frame2 = uint8(frame2);
frame3 = uint8(frame3);
subplot(131)
imshow(frame1)
title('origin')
subplot(132)
imshow(frame2)
title('L')
subplot(133)
imshow(frame3)
title('S')


