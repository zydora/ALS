function accuracy = test(TestImages,W,index,TestLabels)
% A 785*N ; W 256*785 256*257 10*257 ; b 256* 1 256*1 10*1
N = size(TestImages,2);
L = size(W,2);
A = cell(1,L+1);
A{1} = ([ones(N,1) TestImages'])';
for i = 1:L
    if i ~= L
        if index == 1
            A{i+1} = ([ones(N,1) (sigmoid(W{i}*A{i}))'])';
        elseif index == 2
            A{i+1} = ([ones(N,1) (relu(W{i}*A{i}))'])';
        elseif index == 0
            A{i+1} = ([ones(N,1) (W{i}*A{i})'])';
        end
    else
        A{i+1} =(W{i}*A{i});
    end
end
[~,Index] = max(A{L+1});
Index = im2double(Index);
TestLabels = im2double(TestLabels);
accuracy = 1-size(find(Index'-TestLabels-1)~=0,1)/size(TestLabels,1);
end