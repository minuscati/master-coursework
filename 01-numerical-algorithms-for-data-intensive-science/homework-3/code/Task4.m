cd("~/Documents/Education/Master/NumAlgo/HW3/");

% loading the data
load('hw3_zalando.mat');
% correct: vector of labels
% items: matrix of images

% a)
cat = unique(correct); % there are 2 categories
idx1 = find(correct == cat(1), 1); % first image from cat1
idx2 = find(correct == cat(2), 1); % first image from cat2
% plot for cat1
figure;
zalando_plot(items(:, idx1));
title("First Category of Items");
% plot for cat2
figure;
zalando_plot(items(:, idx2));
title('Second Category of Items');
% the two categories are jackets and shoes

% b)
% from hw3_zalando_direct.m:
load 'hw3_zalando.mat';
p=size(items,2); % Nof items
% The data contains exactly
%   p/2 images of item category 4 and
%   p/2 images of item category 9
I=kmeans(items',2); % kmeans returns 1 or 2. But we don't know
                    % which one corresponds to 4 and 9. Try the best
guess1=(I==1)*4+(I==2)*9;
guess2_fft=(I==1)*9+(I==2)*4;
% How many incorrectly classified images?
% total number of incorrectly clustered images
min(sum(guess1 ~= correct ), sum(guess2_fft ~= correct ))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% which cluster is which category?
if sum(guess1 ~= correct) < sum(guess2_fft ~= correct) 
    best = guess1;
else 
    best = guess2_fft;
end
% plot first wrong classification
err = find(best ~= correct);
figure;
zalando_plot(items(:, err(1)));
title("First Incorrectly Classified Image")

% c)
% from hw3_zalando_conv.m:
% Filter to use
zz=[-14:-1, 1:14]; % row  
% alpha=5;
% alpha=10;
alpha = 1;
filter=exp(-zz.^2/alpha).*zz;
n=28;  % size of each image

% Circulant matrix multiplication
function x=matvec_circulant(z,b) % two column vectors as input
    % Naive method
    C=toeplitz(z,[z(1);z(end:-1:2)]); % Create the circulant matrix via a Toeplitz matrix
    x=C*b; % performs the matrix-vector multiplication
end

tic; 
items_fft=zeros(size(items)); % Filtered images

for i=1:size(items,2) % Apply a one-dimensional filter
    X=reshape(items(:,i),n,n); % X a 28x28 matrix with data entry as the i-th column
    Z=zeros(n,n); % Z is a 28x28 matrix 
    for j=1:n
        z=real(matvec_circulant(filter(:) ... % filter now is a column 28x1
            ,X(:,j)));
        Z(:,j)=z; % j-th column with the col-vector computed via the circulant matrix
    end
    items_fft(:,i)=Z(:); % all the columns of Z are stack in the i-th column of items_conv 
end
time_naive = toc; 

I=kmeans(items_fft',2); % kmeans returns 1 or 2. But we don't know
                         % which one corresponds to 4 and 9. Try the best
guess1=(I==1)*4+(I==2)*9;
guess2_fft=(I==1)*9+(I==2)*4;
% How many incorrectly classified images?
min_naive = min(sum(guess1 ~= correct ), sum(guess2_fft ~= correct ));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% which cluster is which category?
if sum(guess1 ~= correct) < sum(guess2_fft ~= correct) 
    best2 = guess1;
else 
    best2 = guess2_fft;
end
% plot first wrong classification
err = find(best2 ~= correct);
figure;
zalando_plot(items(:, err(1)));
title("First Incorrectly Classified Image")

% e)
% Fast algorithm implementation:
function x_hat=matvec_fft(z,b) % two column vectors as input
    z_hat = fft(z);
    b_hat = fft(b);
    x_hat = ifft(z_hat .* b_hat);
end

tic; 
items_fft=zeros(size(items)); % Filtered images

for i=1:size(items,2) % Apply a one-dimensional filter
    X=reshape(items(:,i),n,n); % X a 28x28 matrix with data entry as the i-th column
    Z=zeros(n,n); % Z is a 28x28 matrix 
    for j=1:n
        z=real(matvec_fft(filter(:) ... % filter now is a column 28x1
            ,X(:,j)));
        Z(:,j)=z; % j-th column with the col-vector computed via the circulant matrix
    end
    items_fft(:,i)=Z(:); % all the columns of Z are stack in the i-th column of items_conv 
end
time_fft = toc; 

I=kmeans(items_fft',2); % kmeans returns 1 or 2. But we don't know
                         % which one corresponds to 4 and 9. Try the best
guess1_fft=(I==1)*4+(I==2)*9;
guess2_fft=(I==1)*9+(I==2)*4;
min_fft = min(sum(guess1_fft ~= correct ), sum(guess2_fft ~= correct ));
