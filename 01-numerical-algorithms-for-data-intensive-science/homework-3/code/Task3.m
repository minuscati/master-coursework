%% d)


N=[1000,2000,3000,4000,];
% N=[2000,4000,6000,8000,10000];
% N = [2000,6000,9000,12000,15000];

TEN=zeros(length(N),1);
TEH=zeros(length(N),1);
TEM=zeros(length(N),1);
for i = 1:length(N)
    n = N(i);
    randn('seed',0);
    v=randn(n,1);
    D=diag(randn(n,1));
    a = v*v';
    A=v*v'+D;
    x=randn(n,1);
    
    
    % Build a bisection tree 8,64,256
    stopsize=8; % Do not subdivide intervals smaller than this.
    % You may want to experiment with smaller or larger stopsize to
    % improve performance
    [index_dict,child_dict]=bisection_tree(1,n, stopsize);
    
    display('Computed tree');

    % naive method
    f_naive = @(x) matvec_naive(x, A); %naive product
    tic;
    [V,D]=eigs(f_naive,n,3,'sr'); %compute three eigenvalues
    TEN(i) = toc;

    % optimized holder
    f_smat= @(x) matvec_hodlr(x, A, v, index_dict, child_dict);
    tic;
    [V1,D1] = eigs(f_smat, n, 3, 'sr'); % compute three eigenvalues using HODLR
    TEH(i)=toc;

    % R function
    tic;
    [V2,D2]=eigs(A,3,'sr');
    TEM(i)=toc;

end

TDiff = TEN-TEH;
TDiff2 = TEH-TEM;
% figure;
% plot(N,TDiff,'LineWidth',1.5);
% xlabel('Matrix Size');
% ylabel('Time Difference');
% yline(0,'--r','LineWidth',1); 
