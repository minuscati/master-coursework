%% Task 2_2
n0=5;p=3;
randn('seed',0);
% rng(0,'twister');
c=2;ee=2.5;
A1=randn(n0,p)+c*[1,0,0];
A2=randn(n0,p)+c*[0,1,0];
A3=randn(n0,p)+c*[0,0,1];
A4=[0,0,0];
A=[A1;A2;A3;A4];

% Similarity_graph(A,ee);

%% b
[~,~,E1]=Simi_graph(A,c,ee);



%% c
n0=5; p=3;
randn('seed',0);
c=3;
ee=2.5;
A1=randn(n0,p)+c*[1,0,0];
A2=randn(n0,p)+c*[0,1,0];
A3=randn(n0,p)+c*[0,0,1];
A4=[0,0,0];
A=[A1;A2;A3;A4];
figure;
[~,~,E2]=Simi_graph(A,c,ee);
figure;
Similarity_graph(A,ee);

%% d
n0=5; p=3;
randn('seed',0);
c=3.1;
ee=2.8;
A1=randn(n0,p)+c*[1,0,0];
A2=randn(n0,p)+c*[0,1,0];
A3=randn(n0,p)+c*[0,0,1];
A4=[0,0,0];
A=[A1;A2;A3;A4];
figure;
[L3,V3,E3]=Simi_graph(A,c,ee);

% the second eigenvector
EV2=V3(:,2);
% test
% L3*EV2

%plot
figure
plot(1:16,EV2,'o',LineWidth=1.5);
xlabel('i');
ylabel('v2(i)');

%% e
EV3=V3(:,3);
figure;
plot(EV2,EV3,'o',LineWidth=1.5);
xlabel('v2(i)');
ylabel('v3(i)');