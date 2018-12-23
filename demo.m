%% load dataset
% MINST
clear all;clc
Dataset = 'CIFAR';%You should change the W number nodes as same.
[TrainImages,TestImages,TrainLabels,TestLabels] = load_dataset(Dataset);
%% parameter
N = size(TrainImages,2);
MAXITER = 60;
Lrate = 0.01;
threshold = 1e-3;
batchsize = 1;
% activation mode
index = 1;% case 1:sigmoid activation; case 2:relu activation; case 0:non function
% Matrix completion mode
mc = 0;% case 1:NonCVX_MC % case 2:IHT_MC % case 3:IST_MC % case 0: non MC
% forward update mode
fm = 1;% case 1:update W % case 2:non update
% Layer number
L = 3;

%% Initialize
% [b W]
W = cell(1,L);
W{1} = randn(256,1+size(TrainImages,1))/10;
W{2} = randn(256,1+size(W{1},1))/10;
W{3} = randn(10,1+size(W{2},1))/10;
%W{4} = randn(10,1+128)/10;
%W{5} = randn(10,1+20)/10;
b{1} = zeros(size(W{1},1),1);b{2} = zeros(size(W{2},1),1);b{3} = zeros(size(W{3},1),1);
%b{4} = zeros(10,1);b{5} = zeros(10,1);
for i = 1:L
    W{i}(:,1) = b{i};
end
% [ones A]'
A = cell(1,L+1);
Z = cell(1,L);
A{1} = TrainImages(:,1:N);
A{1} = [ones(N,1) A{1}']';
% Labels
y = zeros(10,N);
    for i = 1:N
        y(TrainLabels(i)+1,i) = 1;
    end
% Init
[A,Z] = forward(A,W,index);
Z{L} = y;

%% ALS + MC
[W,e] = ALS_MC(W,A,Z,y,index,mc,fm,TrainImages,TrainLabels,TestImages,TestLabels,N,MAXITER,threshold);
%% ALS
%[W,e] = ALS(W,A,Z,y,index,mc,fm,TrainImages,TrainLabels,TestImages,TestLabels,N,MAXITER,threshold);
%% BP
%[W,e] = bp(W,TrainImages,y,Lrate,index,TestLabels,TestImages,threshold,MAXITER,batchsize);
%% BP-MC
%[W,e] = bp_mc(W,TrainImages,y,Lrate,index,mc,threshold);
%% ALS + BNN
%[W,e] = ALS_BNN(W,A,Z,y,index,mc,fm,TrainImages,TrainLabels,TestImages,TestLabels,N,MAXITER,threshold);

%% Accuracy for test
accuracy = test(TestImages,W,index,TestLabels);
fprintf(sprintf('The accuracy is %f',accuracy));