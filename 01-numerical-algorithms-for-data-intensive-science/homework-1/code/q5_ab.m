% 5
%% 演示
%%% 读取图片
im=imread('1.png');
size(im);
imshow(im);
im(1,1,:);
%%% 改浮点数
imfloat=double(im);
imfloat(1,1,:);
%%% 换颜色 展示新图片
imfloat(1:100,1:100,1)=255;
imfloat(1:100,1:100,2)=0;
imfloat(1:100,1:100,3)=0;
im_new=uint8(imfloat);
imshow(im_new);
%%% 一张图转变成一个向量
n1=size(imfloat,1); n2=size(imfloat,2); n3=3;
n=n1*n2*n3;
v=reshape(imfloat,n,1);
size(v);
%%% 向量转回图片
img_new2=reshape(v,n1,n2,n3);

%% 练习
for k = 1:27
    filename = sprintf("testbild_snapshots/testbild_snapshots_%04d.png", k);
    frame_k = imread(filename);

    if k == 1
        [H, W, C] = size(frame_k);  
        frame = zeros(H, W, C, 27, 'uint8'); 
    end

    frame(:,:,:,k) = frame_k;  
end


