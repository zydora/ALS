
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
mc = 0;% case 1:NonCVX_MC % case 2:IHT_MC % case 3:IST_MC % case 0: non MC
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
y = -ones(10,N);
for i = 1:N
    y(TrainLabels(i)+1,i) = 1;
end
% Init
[A,Z] = forward(A,W,index);
Z{3} = y;

%% ALS + MC
%[W,e] = ALS_MC(W,A,Z,y,index,mc,fm,TrainImages,TrainLabels,TestImages,TestLabels,N,MAXITER,threshold);
%% ALS
%[W,e] = ALS(W,A,Z,y,index,mc,fm,TrainImages,TrainLabels,TestImages,TestLabels,N,MAXITER,threshold);
%% BP
[W,e] = bp(W,TrainImages,y,Lrate,index,TestLabels,threshold,MAXITER,threshold,batchsize);
%% BP-MC
%[W,e] = bp_mc(W,TrainImages,y,Lrate,index,mc,threshold);

%% Accuracy for test
accuracy = test(TestImages,W,index,TestLabels);
fprintf(sprintf('The accuracy is %f',accuracy));