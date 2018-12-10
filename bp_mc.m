function [W,e] = bp_mc(W,TrainImages,y,Lrate,index,mc,threshold)
while(1)
    N = size(TrainImages,2);
    A = cell(1,4);
    if index == 1
        for i = 1:N
            error = zeros(1,2);
            A{1} = TrainImages(:,i);
            [A,Z] = initforward(A,W,index);
            assistS{3} = diag(ones(size(Z{3})));%10 10
            assistS{2} = diag((ones(257,1)-A{3}).*A{3});% 257 257
            assistS{1} = diag((ones(257,1)-A{2}).*A{2});% 257 257
            e = (y(:,i)-Z{3});% error of the last layer
            S{3} = -2*assistS{3}*e;% 10 1
            S{2} = assistS{2}*(W{3}'*S{3});% 257 1
            S{1} = assistS{1}*(W{2}'*S{2}(2:end));% 256 1
            W{3} = W{3}-Lrate*S{3}*A{3}';% 10 257
            W{2} = W{2}-Lrate*S{2}(2:end)*A{2}';
            W{1} = W{1}-Lrate*S{1}(2:end)*A{1}';
        end
    elseif index == 2
        for i = 1:N
            error = zeros(1,2);
            A{1} = TrainImages(:,i);
            [A,Z] = initforward(A,W,index);
            assistS{3} = diag(ones(size(Z{3})));%10 10
            assistS{2} = diag((ones(257,1)));% 257 257
            assistS{1} = diag((ones(257,1)));% 257 257
            e = (y(:,i)-Z{3});% 10 1
            S{3} = -2*assistS{3}*e;% 10 1
            S{2} = assistS{2}*(W{3}'*S{3});% 257 1
            S{1} = assistS{1}*(W{2}'*S{2}(2:end));% 256 1
            W{3} = W{3}-Lrate*S{3}*A{3}';% 10 257
            W{2} = W{2}-Lrate*S{2}(2:end)*A{2}';
            W{1} = W{1}-Lrate*S{1}(2:end)*A{1}';
            for j = 1:2
                tempA{j+1}(:,i) = A{j+1};
                tempZ{j}(:,i) = Z{j};
                tempA{j+1} = relu(tempA{j+1});
                temp{j} = MC(tempA{j+1}(2:end,:),mc);
                errorZM(j) = norm(temp{j}-Z{j});
                errorAM(j) = norm(temp{j}-tempA{j+1}(2:end));
                fprintf(sprintf('The error between the %d layer Z and M-Z is %f',j,norm(errorZM(j))));
                fprintf(sprintf('The error between the %d layer A and M-Z is %f',j,norm(errorAM(j))));
            end
        end
    end
end
end
