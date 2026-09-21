
function makevideo(A,m,n,fname)
%MAKEVIDEO Make a video from the columns of a snapshot matrix
%   MAKEVIDEO(A,m,n,fname) creates a video (AVI format) from the snapshot
%   matrix A, whose columns contain m by n color images (so that the 
%   columns of A has 3mn elements). fname is the name of the output 
%   video file.

Nb=size(A,2);

outputVideo = VideoWriter(fname);
outputVideo.FrameRate = 4;
open(outputVideo)

for k=1:Nb
    imfloat = reshape(A(:,k),m,n,3);
    writeVideo(outputVideo,uint8(imfloat));
end

close(outputVideo)