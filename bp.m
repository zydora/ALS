function [W,e] = bp(W,TrainImages,y,Lrate,index)
N = size(TrainImages,2);
A = cell(1,4);
if index == 1
for i = 1:N
    A{1} = TrainImages(:,i);
    [A,Z] = forward(A,W,index);
    assistS{3} = diag(ones(size(Z{3})));%10 10
    assistS{2} = diag((ones(257,1)-A{3}).*A{3});% 257 257
    assistS{1} = diag((ones(257,1)-A{2}).*A{2});% 257 257
    e = (y(:,i)-Z{3});% 10 1
    S{3} = -2*assistS{3}*e;% 10 1
    S{2} = assistS{2}*(W{3}'*S{3});% 257 1
    S{1} = assistS{1}*(W{2}'*S{2}(2:end));% 256 1
    W{3} = W{3}-Lrate*S{3}*A{3}';% 10 257
    %b{3} = b{3}-Lrate*S{3};
    W{2} = W{2}-Lrate*S{2}(2:end)*A{2}';
    %b{2} = b{2}-Lrate*S{2};
    W{1} = W{1}-Lrate*S{1}(2:end)*A{1}';
 
    %b{1} = b{1}-Lrate*S{1};
    %sum = sum + norm(e);
end
elseif index == 2
    for i = 1:N
    A{1} = TrainImages(:,i);
    [A,Z] = forward(A,W,index);
    assistS{3} = diag(ones(size(Z{3})));%10 10
    assistS{2} = diag((ones(257,1)));% 257 257
    assistS{1} = diag((ones(257,1)));% 257 257
    e = (y(:,i)-Z{3});% 10 1
    S{3} = -2*assistS{3}*e;% 10 1
    S{2} = assistS{2}*(W{3}'*S{3});% 257 1
    S{1} = assistS{1}*(W{2}'*S{2}(2:end));% 256 1
    W{3} = W{3}-Lrate*S{3}*A{3}';% 10 257
    %b{3} = b{3}-Lrate*S{3};
    W{2} = W{2}-Lrate*S{2}(2:end)*A{2}';
    %b{2} = b{2}-Lrate*S{2};
    W{1} = W{1}-Lrate*S{1}(2:end)*A{1}';
 
    %b{1} = b{1}-Lrate*S{1};
    %sum = sum + norm(e);
    end
end
end