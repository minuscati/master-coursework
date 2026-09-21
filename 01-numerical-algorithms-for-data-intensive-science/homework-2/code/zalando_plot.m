function zalando_plot(z)
% Plotting helper function for the data
%    zalando_items.mat
% which were generated from the training data
% of the Zalando fashion dataset
%
% https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/
%

n=28; % Image size
A=reshape(z,n,n);

% Normalize it
[~,I]=max(abs(z));
za=z(I);
A=A/za;

B=1-A'; % It looks nicer with a white background.

% Plot
imshow(1-A','InitialMagnification',1000)
