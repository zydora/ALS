
%% load dataset
clear all;clc
TrainImages = loadMNISTImages('train-images-idx3-ubyte');
TestImages = loadMNISTImages('t10k-images-idx3-ubyte');
TrainLabels = loadMNISTLabels('train-labels-idx1-ubyte');
TestLabels = loadMNISTLabels('t10k-labels-idx1-ubyte');

%% parameter
N = 60000;
MAXITER = 60;
Lrate = 0.01;
threshold = 1e-3;
% activation mode
index = 2;% case 1:sigmoid activation; case 2:relu activation
% Matrix completion mode
mc = 2;% case 1:NonCVX_MC % case 2:IHT_MC % case 3:IST_MC
% forward update mode
fm = 2;% case 1:update W % case 2:non update

%% Initialize
% [b W]
W = cell(1,3);
W{1} = randn(256,1+784)/10;
W{2} = randn(256,1+256)/10;
W{3} = randn(10,1+256)/10;
b{1} = zeros(256,1);b{2} = zeros(256,1);b{3} = zeros(10,1);
for i = 1:3
    W{i}(:,1) = b{i};
end
% [ones A]'
A = cell(1,4);
Z = cell(1,3);
A{1} = TrainImages(:,1:N);
A{1} = [ones(N,1) A{1}']';
% Labels
y = zeros(10,N);
for i = 1:N
    y(TrainLabels(i)+1,i) = 1;
end
% Init
[A,Z] = forward(A,W,index);
Z{3} = y;

%% ALS + MC
[W,e] = ALS_MC(W,A,Z,y,index,mc,fm,TrainImages,TrainLabels,TestImages,TestLabels,N,MAXITER);
%% ALS
[W,e] = ALS(W,A,Z,y,index,mc,fm,TrainImages,TrainLabels,TestImages,TestLabels,N,MAXITER);
%% BP
[W,e] = bp(W,TrainImages,y,Lrate,index,TestLabels,threshold,MAXITER);

%% Accuracy for test
M = cell(1,4);M{1} = TestImages;
[M,~] = initforward(M,W,index);
[~,Index] = max(M{4});
accuracy = 1-size(find(Index'-TestLabels-1)~=0,1)/10000