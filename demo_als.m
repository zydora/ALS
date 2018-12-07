% ALS
%% load dataset
clear all;clc
TrainImages = loadMNISTImages('train-images-idx3-ubyte');
TestImages = loadMNISTImages('t10k-images-idx3-ubyte');
TrainLabels = loadMNISTLabels('train-labels-idx1-ubyte');
TestLabels = loadMNISTLabels('t10k-labels-idx1-ubyte');

%% parameter
N = 60000;
MAXITER = 60;
%Lrate = 0.01;
threshold = 1e-3;
index = 1;
mc = 1;

%% Initialize
%W b
W = cell(1,3);
W{1} = randn(256,1+784)/10;
W{2} = randn(256,1+256)/10;
W{3} = randn(10,1+256)/10;
b{1} = zeros(256,1);b{2} = zeros(256,1);b{3} = zeros(10,1);
for i = 1:3
    W{i}(:,1) = b{i};
end
% A Z
A = cell(1,4);
Z = cell(1,3);
A{1} = TrainImages;
% Labels
y = zeros(10,N);
for i = 1:N
    y(TrainLabels(i)+1,i) = 1;
end

[A,Z] = initforward(A,W,index);
Z{3} = y;

%% Iteration
normt = 0;
itr = 1;
while (1 )
    %sweep from ltr
    for i = 1:3
            %E = eye(size(A{i},1));
            %E(1,1) = 0;
            W{i} = Z{i}/A{i};
            %Z{i} = W{i}*A{i};
            if i~= 3
            A{i+1} = ([ones(N,1) sigmoid(W{i}*A{i})'])';
            end
    end
    %sweep from rtl
    for i = 3:-1:1
        E = eye(size(A{i},1));
        W{i} = Z{i}/A{i};
        if i ~=3
            if index == 1
                %temp = min(0,max(1,A{i+1}(2:end,:)));
                temp = A{i+1}(2:end,:);
                temp = MC(temp,mc);
                Z{i} = log(temp./(1-temp));   
            elseif index == 2
                temp = relu(A{i+1}(2:end,:));
                Z{i} = MC(temp,mc);
            end
        end
    end
    e = norm(y-W{3}*A{3})/N;
    Output = test(TestImages,W,index);
    [~,Index] = max(Output);
    accuracy = 1-size(find(Index'-TestLabels-1)~=0,1)/10000;
    fprintf('')
    fprintf(sprintf(' %itr iterations ',itr));
    fprintf(sprintf(' accuracy : %f\n',accuracy));
    itr = itr + 1;
    if itr > MAXITER || e< threshold
        break;
    end
end

%% Accuracy for test
M = cell(1,4);M{1} = TestImages;
[M,~] = initforward(M,W,index);
[~,Index] = max(M{4});
accuracy = 1-size(find(Index'-TestLabels-1)~=0,1)/10000
