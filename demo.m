%% load dataset
clear all;clc
Dataset = 'MNIST';%You should change the W number nodes as same.
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
% Layer number
L = 5; 

%% Initialize
% [b W]
W = cell(1,L);num = 600;
W{1} = randn(num,1+size(TrainImages,1))/10;
for i = 2:L-1
W{i} = randn(num,1+size(W{i-1},1))/10;
end
W{L} = randn(10,1+size(W{L-1},1))/10;
for i = 1:L
b{i} = zeros(size(W{i},1),1);
end
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
%[W,e] = ALS_MC(W,A,Z,y,index,mc,TrainImages,TrainLabels,TestImages,TestLabels,N,MAXITER,threshold);
%% BP
[W,e] = bp(W,TrainImages,y,Lrate,index,TestLabels,TestImages,threshold,MAXITER,batchsize);
%% BP-MC
%[W,e] = bp_mc(W,TrainImages,y,Lrate,index,mc,threshold);
%% ALS + BNN
%[W,e] = ALS_BNN(W,A,Z,y,index,mc,TrainImages,TrainLabels,TestImages,TestLabels,N,MAXITER,threshold);

%% Accuracy for test
accuracy = test(TestImages,W,index,TestLabels);
fprintf(sprintf('The accuracy is %f',accuracy));