% main function for BP %whole structure instead of 10 class verify
%clear all;clc;
TrainImages = loadMNISTImages('train-images-idx3-ubyte');
TestImages = loadMNISTImages('t10k-images-idx3-ubyte');
TrainLabels = loadMNISTLabels('train-labels-idx1-ubyte');
TestLabels = loadMNISTLabels('t10k-labels-idx1-ubyte');
%% parameter
N = 60000;
Lrate = 0.06;
threshold = 1e-2;
%Initialize
W = cell(1,3);
W{1} = randn(256,1+784)/10;
W{2} = randn(256,1+256)/10;
W{3} = randn(10,1+256)/10;
b{1} = zeros(256,1);b{2} = zeros(256,1);b{3} = zeros(10,1);
for i = 1:3
    W{i}(:,1) = b{i};
end
Z = cell(1,3);
y = zeros(10,N);%label one-hot
for i = 1:N
    y(TrainLabels(i)+1,i) = 1;
end
%% Iteration
%initialize
itr = 1;
normt = 0;
tic
while (1)
    [W,e] = bp(W,TrainImages,y,Lrate,1);
%     M = cell(1,4);
%     M{1} = TestImages;
%     [M,~] = forward(M,W,1);
%     [~,Index] = max(M{4});
%     accuracy = 1-size(find(Index'-TestLabels-1)~=0,1)/10000;
    fprintf('')
    fprintf(sprintf('%itr iterations\n',itr));
    fprintf(sprintf('error : %f\n %f',norm(e),accuracy));
    itr = itr + 1;
    normt = [norm(e) normt];
    if norm(e)<threshold
        break;
    end
    if normt(1)>normt(2)
        Lrate = Lrate/2;
    end
end
toc
%% Accuracy for test
Output = test(TestImages,W);
[~,Index] = max(Output);
accuracy = 1-size(find(Index'-TestLabels-1)~=0,1)/10000;
%% Results
fprintf(sprintf('For learning rate %f and %itr iterations\n',Lrate,itr));
fprintf(sprintf('the accuracy for test set is %f',accuracy));