function accuracy = test(TestImages,W,index,TestLabels)
% A 785*N ; W 256*785 256*257 10*257 ; b 256* 1 256*1 10*1
N = size(TestImages,2);
A = cell(1,4);
A{1} = ([ones(N,1) TestImages'])';
for i = 1:3
    if i == 1|| i == 2
        if index == 1
        A{i+1} = ([ones(N,1) (sigmoid(W{i}*A{i}))'])';
        elseif index == 2
            A{i+1} = ([ones(N,1) (relu(W{i}*A{i}))'])';
        end
    else
        A{i+1} =(W{i}*A{i});
        %A4 = W{3}*A3+repmat(b{3},N,2);
    end
end
[~,Index] = max(A{4});
accuracy = 1-size(find(Index'-TestLabels-1)~=0,1)/size(TestLabels,1);
